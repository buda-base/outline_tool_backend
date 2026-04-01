"""
TEI/XML to standoff text conversion module.

This module provides functionality to convert TEI/XML format documents to plain text
with standoff annotations. It has minimal dependencies (only lxml and standard library).

Main API:
    text, annotations, source_path = convert_tei_root_to_standoff(tree)
"""

import logging
import re
from bisect import bisect
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from lxml import etree

logger = logging.getLogger(__name__)


def replace_element(
    old_element: etree._Element,
    new_element: etree._Element | None = None,
) -> None:
    """
    Replace or remove an XML element while preserving content structure.

    Args:
        old_element: The element to replace or remove
        new_element: The replacement element, or None to remove without replacement
    """
    parent = old_element.getparent()
    if parent is None:
        raise ValueError("Cannot replace/remove the root element")

    # Save the tail text from the old element
    tail_text = old_element.tail

    if new_element is None:
        # REMOVAL CASE
        # Find the node where we should append the tail text
        prev_sibling = old_element.getprevious()

        # Remove the old element
        parent.remove(old_element)

        # Handle the tail text
        if tail_text:
            if prev_sibling is not None:
                # Append to previous sibling's tail
                if prev_sibling.tail:
                    prev_sibling.tail += tail_text
                else:
                    prev_sibling.tail = tail_text
            # Append to parent's text
            elif parent.text:
                parent.text += tail_text
            else:
                parent.text = tail_text
    else:
        # REPLACEMENT CASE
        # Set the tail on the new element
        new_element.tail = tail_text

        # Perform the replacement
        parent.replace(old_element, new_element)


def add_position_diff(
    positions: list[int],
    diffs: list[int],
    position: int,
    cumulative_diff: int,
) -> None:
    """Track position differences during text transformation."""
    if not positions or position > positions[-1]:
        positions.append(position)
        diffs.append(cumulative_diff)
    else:
        # case where we overwrite the latest diff
        diffs[-1] = cumulative_diff


def correct_position(current_position: int, positions: list[int], diffs: list[int]) -> int:
    """Correct a position based on accumulated diffs."""
    previous_position_i = bisect(positions, current_position)
    if previous_position_i < 1:
        return current_position
    return current_position + diffs[previous_position_i - 1]


def apply_position_diffs(
    positions: list[int],
    diffs: list[int],
    annotations: dict[str, Any],
    collapsed_spans: list[tuple[int, int, int]] | None = None,
) -> None:
    """
    Apply position diffs to annotations.
    For positions that were inside a SHRUNK match:
      - preserve relative offset if it still fits in the replacement,
      - otherwise clamp to the end of the replacement.
    """
    collapsed_spans = collapsed_spans or []

    def remap_if_inside_shrunk(orig_pos: int) -> int | None:
        """
        If orig_pos was inside a shrinking match [s, e),
        map it into the output using the replacement length:
          new_pos = out_start + min(rel, replacement_len)
        where out_start is the output idx corresponding to original s.
        """
        for s, e, repl_len in collapsed_spans:
            if s <= orig_pos < e:
                rel = orig_pos - s
                # anchor at start (in output space), then add clamped rel
                out_start = correct_position(s, positions, diffs)
                # clamp rel to replacement length (end is valid landing spot)
                clamped_rel = min(repl_len, rel)
                return out_start + clamped_rel
        return None  # not inside any shrunk span

    def adjust_point(orig_pos: int) -> int:
        mapped = remap_if_inside_shrunk(orig_pos)
        if mapped is not None:
            return mapped
        # default path: use the normal diff-based correction
        return correct_position(orig_pos, positions, diffs)

    for ann_type, ann_list in annotations.items():
        if ann_type == "milestones":
            # dict: id -> coordinate
            for milestone_id in ann_list:
                ann_list[milestone_id] = adjust_point(ann_list[milestone_id])
        else:
            # list of dicts with cstart/cend
            for ann in ann_list:
                ann["cstart"] = adjust_point(ann["cstart"])
                ann["cend"] = adjust_point(ann["cend"])


def get_string(
    orig: str,
    pattern_string: str,
    repl_fun: Callable[[re.Match, int], str],
    annotations: dict[str, Any],
) -> str:
    """
    Apply regex replacement to string while tracking position changes for annotations.
    For shrinking matches, also record spans and replacement sizes so we can remap
    points that were originally inside the match.

    Args:
      orig: Original string
      pattern_string: Regex pattern
      repl_fun: Replacement function that takes (match, output_len) and returns replacement
      annotations: Annotations dict to update with position diffs

    Returns: Transformed string
    """
    p = re.compile(pattern_string, flags=re.MULTILINE | re.DOTALL)
    diffs = []
    positions = []
    output_parts: list[str] = []
    output_len = 0
    cumulative = 0
    last_match_end = 0

    # (start, end, replacement_len) for shrinking matches
    collapsed_spans = []

    for m in p.finditer(orig):
        group_size = m.end() - m.start()
        skipped_size = m.start() - last_match_end
        output_parts.append(orig[last_match_end : m.start()])
        last_match_end = m.end()
        output_len += skipped_size

        replacement = repl_fun(m, output_len)
        replacement_len = len(replacement)

        if replacement_len < group_size:
            # record shrinking span for smart point remapping later
            collapsed_spans.append((m.start(), m.end(), replacement_len))

            ot_len = 0
            if "ot" in m.groupdict():  # keep your existing special-case
                ot_len = len(m.group("ot"))
                add_position_diff(positions, diffs, m.start() + 1, cumulative - ot_len)

            cumulative += replacement_len - group_size
            add_position_diff(positions, diffs, m.end(), cumulative)

        elif replacement_len > group_size:
            # when the replacement is large, new indexes point to
            # the last original index
            for i in range(group_size, replacement_len):
                cumulative -= 1
                add_position_diff(positions, diffs, output_len + i, cumulative)

        output_parts.append(replacement)
        output_len += replacement_len

    if last_match_end == 0:
        # no match
        return orig

    if last_match_end < len(orig):
        output_parts.append(orig[last_match_end:])

    # pass collapsed spans so we can remap interior points
    apply_position_diffs(positions, diffs, annotations, collapsed_spans=collapsed_spans)
    return "".join(output_parts)


def convert_pages(text: str, annotations: dict[str, Any]) -> str:
    """Replace <pb_marker>{pname}</pb_marker> or <pb_marker/> with spacing and track page boundaries."""
    page_annotations: list[dict[str, Any]] = []

    def repl_pb_marker(m: re.Match, cstart: int) -> str:
        # Handle both <pb_marker>text</pb_marker> and <pb_marker/>
        pname = m.group("pname") or ""
        # pname can be None or empty string if <pb/> has no n attribute
        # Only include pname in the annotation if it has a value
        page_ann: dict[str, Any] = {"cstart": cstart + 2 if cstart > 0 else 0}
        if pname:
            page_ann["pname"] = pname
        page_annotations.append(page_ann)
        # don't replace the first one
        return "\n\n" if cstart > 0 else ""

    # Match both <pb_marker>text</pb_marker> and self-closing <pb_marker/>
    pat_str = r"[\r\n\s]*(?:<pb_marker>(?P<pname>.*?)</pb_marker>|<pb_marker\s*/>)[\r\n\s]*"
    output = get_string(text, pat_str, repl_pb_marker, annotations)
    for i, p_ann in enumerate(page_annotations):
        p_ann["pnum"] = i + 1
        # assert that the first page starts at the beginning
        if i < len(page_annotations) - 1:
            p_ann["cend"] = page_annotations[i + 1]["cstart"] - 2
        else:
            p_ann["cend"] = len(output)
    annotations["pages"] = page_annotations
    return output


def convert_milestones(text: str, annotations: dict[str, Any]) -> str:
    """
    Replace <milestone_marker>{id}</milestone_marker> with empty string
    and track the milestone coordinates in annotations.
    Only consumes leading whitespace to avoid eating spacing between elements.
    """
    milestone_coords: dict[str, int] = {}

    def repl_milestone_marker(m: re.Match, cstart: int) -> str:
        milestone_id = m.group("id")
        milestone_coords[milestone_id] = cstart
        return ""

    # Only consume leading whitespace, not trailing
    pat_str = r"<milestone_marker>(?P<id>.*?)</milestone_marker>"
    output = get_string(text, pat_str, repl_milestone_marker, annotations)
    if milestone_coords:
        annotations["milestones"] = milestone_coords
    return output


def convert_div_boundaries(text: str, annotations: dict[str, Any]) -> str:
    """
    Replace <div_start_marker/> and <div_end_marker/> with empty strings
    and track div boundaries for chunking.
    Div end markers add spacing to separate adjacent divs.
    """
    div_boundaries: list[dict[str, int]] = []
    current_div_index = -1

    def repl_div_marker(m: re.Match, cstart: int) -> str:
        nonlocal current_div_index
        marker = m.group(0)
        if "div_start_marker" in marker:
            div_boundaries.append({"cstart": cstart, "cend": -1})
            current_div_index += 1
            return ""  # No spacing needed at div start
        if "div_end_marker" in marker:
            if 0 <= current_div_index < len(div_boundaries):
                # Record the position before adding spacing
                div_boundaries[current_div_index]["cend"] = cstart
            return "\n\n"  # Add spacing after div end to separate adjacent divs
        return ""

    # Remove both markers in one pass
    pat_str = r"<div_(start|end)_marker\s*/>"
    output = get_string(text, pat_str, repl_div_marker, annotations)

    # Filter out any incomplete boundaries
    div_boundaries = [b for b in div_boundaries if b["cend"] != -1]

    if div_boundaries:
        annotations["div_boundaries"] = div_boundaries
    return output


def convert_hi(text: str, annotations: dict[str, Any]) -> str:
    """Replace <hi_{rend}>{content}</hi_{rend}> with {content} and save annotation coordinates."""
    if "hi" not in annotations:
        annotations["hi"] = []
    hi_annotations = annotations["hi"]

    def repl_hi_marker(m: re.Match, _cstart: int) -> str:
        rend = m.group("rend")
        hi_annotations.append({"rend": rend, "cstart": m.start(), "cend": m.end()})
        return m.group("content")

    pat_str = r"(?P<ot><hi_(?P<rend>[^>]+)>)(?P<content>.*?)</hi_(?P=rend)>"
    return get_string(text, pat_str, repl_hi_marker, annotations)


def remove_other_markers(text: str, annotations: dict[str, Any]) -> str:
    """Remove all remaining XML markers."""

    def repl_xml_marker(_m: re.Match, _cstart: int) -> str:
        return ""

    pat_str = r"</?[^>]*?>"
    return get_string(text, pat_str, repl_xml_marker, annotations)


def normalize_new_lines(text: str, annotations: dict[str, Any]) -> str:
    """Normalize newlines by removing surrounding whitespace."""

    def repl_nl_marker(_m: re.Match, _cstart: int) -> str:
        return "\n"

    def repl_nl_marker_multi(_m: re.Match, _cstart: int) -> str:
        return "\n\n"

    pat_str = r"[\t \r]*\n[\t \r]*"
    text = get_string(text, pat_str, repl_nl_marker, annotations)
    pat_str = r"\n{3,}"
    return get_string(text, pat_str, repl_nl_marker_multi, annotations)


def unescape_xml(text: str, annotations: dict[str, Any]) -> str:
    """Unescape XML entities."""
    # Common character entities
    simple_replacements = {"&quot;": '"', "&apos;": "'", "&lt;": "<", "&gt;": ">", "&amp;": "&"}

    def repl_esc_xml(m: re.Match, _cstart: int) -> str:
        escaped_entity = m.group(0)
        if escaped_entity in simple_replacements:
            return simple_replacements[escaped_entity]
        num_str = m.group(1)  # e.g. "#123" or "#x2F"
        if num_str.startswith("#x"):
            return chr(int("0x" + num_str.removeprefix("#x"), 0))
        return chr(int(num_str.removeprefix("#")))

    pat_str = r"&(quot|apos|lt|gt|amp|#x[0-9a-fA-F]+|#\d+);"
    return get_string(text, pat_str, repl_esc_xml, annotations)


def _shift_all_annotations(annotations: dict[str, Any], offset: int) -> None:
    """
    Shift all character coordinates by offset in place.
    Handles both list-based annotations, milestone dict, and div_boundaries.

    Args:
        annotations: The annotations dict to modify
        offset: The amount to shift coordinates (can be negative)
    """
    if not offset:
        return

    for key, anno_list in annotations.items():
        if key == "milestones":
            # Milestones is a dict of id -> coordinate
            for milestone_id in anno_list:
                new_pos = anno_list[milestone_id] + offset
                # Clamp to 0 if negative
                anno_list[milestone_id] = max(0, new_pos)
        else:
            # Regular annotations are lists of dicts with cstart/cend
            for anno in anno_list:
                if "cstart" in anno:
                    anno["cstart"] = max(0, anno["cstart"] + offset)
                if "cend" in anno:
                    anno["cend"] = max(0, anno["cend"] + offset)


def align_div_milestones_nl(text: str, annotations: dict[str, Any]) -> None:
    """
    Adjust div boundaries to align with milestones and skip trailing newlines.

    This is a postprocessing step that ensures div boundaries properly align with
    milestones. The issue is that during the multi-step text transformation process,
    position tracking can become misaligned due to tag removal, whitespace normalization,
    etc. This function corrects the div boundaries as a final step.

    For each div, we adjust both cend and the next div's cstart to align with any
    milestones that fall between them. This ensures that milestones properly mark
    div boundaries.

    Args:
        text: The final text string
        annotations: The annotations dict with div_boundaries and milestones
    """
    div_boundaries = annotations.get("div_boundaries")
    milestones = annotations.get("milestones")

    if not div_boundaries or not milestones:
        return

    def _skip_newlines(position: int) -> int:
        while position < len(text) and text[position] == "\n":
            position += 1
        return position

    # Adjust milestones to skip newline characters
    for milestone_id, coord in milestones.items():
        milestones[milestone_id] = _skip_newlines(coord)

    # Adjust div boundaries (cend) to skip newline characters
    for boundary in div_boundaries:
        if "cend" in boundary and boundary["cend"] is not None:
            boundary["cend"] = _skip_newlines(boundary["cend"])


def synthesize_page_boundary_milestones(annotations: dict[str, Any]) -> None:
    """
    Synthesize milestone entries for page boundaries (bop: and eop: markers).

    This creates synthetic milestones based on page annotations so that outline
    content locations can reference page boundaries using:
    - bop:N - beginning of page N (at page cstart)
    - eop:N - end of page N (at page cend)

    These are added to the milestones dict alongside any XML-defined milestones.

    Args:
        annotations: The annotations dict with pages (and optionally milestones)
    """
    pages = annotations.get("pages", [])
    if not pages:
        return

    # Ensure milestones dict exists
    if "milestones" not in annotations:
        annotations["milestones"] = {}

    milestones = annotations["milestones"]

    for page in pages:
        pnum = page.get("pnum")
        if pnum is None:
            continue

        # Create bop:N (beginning of page) at cstart
        if "cstart" in page:
            bop_id = f"bop:{pnum}"
            milestones[bop_id] = page["cstart"]

        # Create eop:N (end of page) at cend
        if "cend" in page:
            eop_id = f"eop:{pnum}"
            milestones[eop_id] = page["cend"]


def _format_context_snippet(text: str, position: int, marker: str, radius: int = 10) -> str:
    """Return a snippet of text around position with marker inserted."""
    position = max(0, min(len(text), position))
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    before = text[start:position]
    after = text[position:end]
    snippet = f"{before}{marker}{after}"
    return snippet.replace("\n", "\\n")


def _debug_log_annotations(text: str, annotations: dict[str, Any]) -> None:
    """Emit detailed debug logs for milestones and div boundaries."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    milestones = annotations.get("milestones", {})
    if milestones:
        logger.debug("Milestones (%d entries):", len(milestones))
        for milestone_id, coord in sorted(milestones.items(), key=lambda item: item[1]):
            marker = f'<id="{milestone_id}">'  # no closing tag per requirements
            snippet = _format_context_snippet(text, coord, marker)
            logger.debug("  %s at %d -> %s", milestone_id, coord, snippet)
    else:
        logger.debug("Milestones: none found")

    divs = annotations.get("div_boundaries", [])
    if divs:
        logger.debug("Div boundaries (%d entries):", len(divs))
        for idx, div in enumerate(divs, start=1):
            cstart = div.get("cstart", 0)
            cend = div.get("cend", 0)
            start_marker = f"<div_{idx}_start>"
            end_marker = f"<div_{idx}_end>"
            start_snippet = _format_context_snippet(text, cstart, start_marker)
            end_snippet = _format_context_snippet(text, cend, end_marker)
            logger.debug("  Div %d start %d -> %s", idx, cstart, start_snippet)
            logger.debug("  Div %d end %d -> %s", idx, cend, end_snippet)
    else:
        logger.debug("Div boundaries: none found")


def trim_text_and_adjust_annotations(text: str, annotations: dict[str, Any]) -> str:
    """
    Remove leading and trailing whitespace from text and adjust annotation coordinates.

    Args:
        text: The text string to trim
        annotations: The annotations dict to adjust

    Returns:
        Trimmed text string
    """
    # Calculate how much we're trimming from the beginning
    leading_match = re.match(r"^[\s\n]*", text)
    s_count = len(leading_match.group()) if leading_match else 0

    # Trim leading whitespace
    if s_count > 0:
        # Shift annotations by negative offset (subtract the trimmed amount)
        _shift_all_annotations(annotations, -s_count)
        text = text[s_count:]

    # Trim trailing whitespace
    e_match = re.search(r"[\s\n]*$", text)
    if e_match and e_match.group():
        e_count = len(e_match.group())
        if e_count > 0:
            # New text length after trimming
            new_length = len(text) - e_count
            text = text[:new_length]

            # Adjust any annotations that extend beyond the new end
            # (though this shouldn't normally happen)
            for key, anno_list in annotations.items():
                if key == "milestones":
                    # Milestones are a dict of id -> coordinate
                    for milestone_id in anno_list:
                        anno_list[milestone_id] = min(anno_list[milestone_id], new_length)
                else:
                    # Regular annotations are lists of dicts with cstart/cend
                    for anno in anno_list:
                        if "cstart" in anno and anno["cstart"] > new_length:
                            anno["cstart"] = new_length
                        if "cend" in anno and anno["cend"] > new_length:
                            anno["cend"] = new_length

    return text


def debug_annotations(text: str, annotations: dict[str, Any]) -> str:
    """Create a debug view of text with annotation boundaries marked."""
    boundaries: list[tuple[int, str]] = []

    for anno_type, anno_list in annotations.items():
        if anno_type == "milestones":
            for milestone_id, milestone_coord in anno_list.items():
                boundaries.append((milestone_coord, f"[milestone_{milestone_id}/]"))
            continue
        for anno in anno_list:
            boundaries.append((anno["cstart"], f"[{anno_type}]"))
            boundaries.append((anno["cend"], f"[/{anno_type}]"))

    # Sort boundaries by position (ascending) and build result in one pass
    boundaries.sort()

    parts: list[str] = []
    last_pos = 0
    for position, marker in boundaries:
        parts.append(text[last_pos:position])
        parts.append(marker)
        last_pos = position
    parts.append(text[last_pos:])
    return "".join(parts)


def convert_tei_root_to_standoff(
    root: etree._Element,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """
    Convert a TEI/XML file to plain text with standoff annotations.

    Conversion rules:
    - Only content within the body tags is processed
    - Line breaks (<lb/>) become newline characters
    - Page breaks (<pb/>) become two newline characters
    - Notes (<note>...</note>) are removed
    - <figure><caption><p>foo</p></caption></figure> becomes "foo"
    - <gap /> elements are removed
    - <unclear><supplied>foo</supplied></unclear> becomes "foo"
    - <choice><orig>foo</orig><corr>bar</corr></choice> becomes "bar"
    - Milestones are tracked in annotations but removed from text
    - Head elements are converted to hi annotations with rend='head'
    - All other XML tags are stripped
    - XML-encoded characters (&gt;, etc.) are converted to their normal representation

    Args:
        root: an etree root element

    Returns:
        tuple: (text, annotations, source_path)
            - text: String containing the plain text representation
            - annotations: Dict representing the annotations
            - source_path: The path of the source file (or None)
    """
    # Find the body element (handle TEI namespace if present)
    tei_ns = "http://www.tei-c.org/ns/1.0"

    source_path_el = root.find(f".//{{{tei_ns}}}idno[@type='src_path']")
    source_path: str | None = source_path_el.text if source_path_el is not None else None

    body_el = root.find(f".//{{{tei_ns}}}body")

    if body_el is None:
        logger.error("No body element found in the TEI document")
        return None, None, None

    # Check if xml:space="preserve" is present
    xml_space_preserve = body_el.get("{http://www.w3.org/XML/1998/namespace}space") == "preserve"

    # Create a deep copy of the body to avoid modifying the original tree
    body_copy = deepcopy(body_el)
    body_copy.tag = "body"

    # Process the TEI elements

    # Handle div elements - mark boundaries for chunking if not xml:space="preserve"
    if not xml_space_preserve:
        for div in body_copy.findall(f".//{{{tei_ns}}}div"):
            # Add markers to track div boundaries
            div_start_marker = etree.Element("div_start_marker")
            div_end_marker = etree.Element("div_end_marker")

            # Insert start marker as first child
            if len(div) > 0:
                div.insert(0, div_start_marker)
            else:
                div_start_marker.text = div.text or ""
                div.text = ""
                div.append(div_start_marker)

            # Append end marker as last child
            div.append(div_end_marker)

    # Handle milestone elements - convert to markers for coordinate tracking
    for milestone in body_copy.findall(f".//{{{tei_ns}}}milestone"):
        milestone_id = milestone.get("{http://www.w3.org/XML/1998/namespace}id")
        if milestone_id:
            milestone_marker = etree.Element("milestone_marker")
            milestone_marker.text = milestone_id
            replace_element(milestone, milestone_marker)
        else:
            replace_element(milestone, None)

    # Handle head elements - convert to hi_head for annotation tracking
    for head in body_copy.findall(f".//{{{tei_ns}}}head"):
        new_tag = etree.Element("hi_head")
        new_tag.text = head.text
        for child in head:
            new_tag.append(deepcopy(child))
        new_tag.tail = head.tail
        replace_element(head, new_tag)

    # Remove all note elements
    for note in body_copy.findall(f".//{{{tei_ns}}}note"):
        replace_element(note, None)

    # Remove all gap elements
    for gap in body_copy.findall(f".//{{{tei_ns}}}gap"):
        text_element = etree.Element("text_marker")
        text_element.text = "X"
        replace_element(gap, text_element)

    # Process figure elements - extract caption text
    for figure in body_copy.findall(f".//{{{tei_ns}}}figure"):
        caption_parts: list[str] = []
        for caption_el in figure.findall(f".//{{{tei_ns}}}caption"):
            for fragment in caption_el.itertext():
                text_fragment = str(fragment).strip()
                if text_fragment:
                    caption_parts.append(text_fragment)
        caption_text = " ".join(caption_parts)

        text_element = etree.Element("text_marker")
        text_element.text = caption_text
        replace_element(figure, text_element)

    for hi in body_copy.findall(f".//{{{tei_ns}}}hi"):
        render_val = hi.get("rend", "")
        # Create new element with the format hi_xxx
        new_tag = etree.Element(f"hi_{render_val}")
        # Copy the text content
        new_tag.text = hi.text
        # Copy all child elements
        for child in hi:
            new_tag.append(deepcopy(child))
        # Copy any tail text
        new_tag.tail = hi.tail
        replace_element(hi, new_tag)

    # Process unclear/supplied elements - keep supplied text
    for unclear in body_copy.findall(f".//{{{tei_ns}}}unclear"):
        unclear_text = "".join(str(t) for t in unclear.itertext())
        text_element = etree.Element("hi_unclear")
        text_element.text = unclear_text
        replace_element(unclear, text_element)

    # Process choice elements - use corr/reg instead of orig/sic
    for choice in body_copy.findall(f".//{{{tei_ns}}}choice"):
        replacement = choice.find(f".//{{{tei_ns}}}corr")
        if replacement is None:
            replacement = choice.find(f".//{{{tei_ns}}}reg")
        if replacement is not None:
            replacement_text = "".join(str(t) for t in replacement.itertext())
            text_element = etree.Element("text_marker")
            text_element.text = replacement_text
            replace_element(choice, text_element)

    # Replace all pb elements with custom markers
    for pb in body_copy.findall(f".//{{{tei_ns}}}pb"):
        pb_marker = etree.Element("pb_marker")
        pnum = pb.get("n")
        if pnum:
            pb_marker.text = pnum
        replace_element(pb, pb_marker)

    # Replace all lb elements with custom markers
    for lb in body_copy.findall(f".//{{{tei_ns}}}lb"):
        lb_marker = etree.Element("lb_marker")
        lb_marker.text = "\n"
        replace_element(lb, lb_marker)

    # Get the text content — strip namespace prefixes so the serialized
    # XML only contains bare tag names that downstream regex passes expect.
    etree.cleanup_namespaces(body_copy)
    xml_str = etree.tostring(body_copy, encoding="unicode", method="xml", pretty_print=False)

    # Simple substitutions
    xml_str = xml_str.replace("\ufeff", "")
    # Handle div and p tags based on xml:space attribute
    if xml_space_preserve:
        # Old behavior: just remove tags, normalize spaces at beginning and end
        xml_str = re.sub(r"[\r\n\t ]*</?(?:body|p|div)(?: +[^>]+)*>[\r\n\t ]*", "", xml_str, flags=re.DOTALL)
    else:
        # New behavior: add two line breaks around divs and ps
        xml_str = re.sub(r"<div(?: +[^>]+)*>", "\n\n", xml_str, flags=re.DOTALL)
        xml_str = re.sub(r"</div>", "\n\n", xml_str, flags=re.DOTALL)
        xml_str = re.sub(r"<p(?: +[^>]+)*>", "\n\n", xml_str, flags=re.DOTALL)
        xml_str = re.sub(r"</p>", "\n\n", xml_str, flags=re.DOTALL)
        xml_str = re.sub(r"</?body(?: +[^>]+)*>", "", xml_str, flags=re.DOTALL)
    xml_str = re.sub(r"(?:\n\s*)?<text_marker>(.*?)</text_marker>(?:\n\s*)?", r"\1", xml_str, flags=re.DOTALL)
    xml_str = re.sub(r"[\r\n\t ]*<lb_marker>(.*?)</lb_marker>[\r\n\t ]*", r"\1", xml_str, flags=re.DOTALL)

    annotations: dict[str, Any] = {}
    xml_str = convert_div_boundaries(xml_str, annotations)
    xml_str = convert_milestones(xml_str, annotations)
    xml_str = convert_pages(xml_str, annotations)
    xml_str = convert_hi(xml_str, annotations)
    xml_str = remove_other_markers(xml_str, annotations)
    xml_str = unescape_xml(xml_str, annotations)
    xml_str = normalize_new_lines(xml_str, annotations)

    # Trim leading and trailing whitespace and adjust annotations
    xml_str = trim_text_and_adjust_annotations(xml_str, annotations)

    # Align div boundaries with milestones (postprocessing)
    if not xml_space_preserve:
        align_div_milestones_nl(xml_str, annotations)

    # Synthesize bop: and eop: milestones for page boundaries
    # These allow outline content locations to reference page boundaries
    synthesize_page_boundary_milestones(annotations)

    _debug_log_annotations(xml_str, annotations)

    return xml_str, annotations, source_path


def convert_tei_to_standoff(
    xml_file_path: str,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """
    Convert a TEI/XML file to plain text with standoff annotations.

    Args:
        xml_file_path: Path to the XML file

    Returns:
        tuple: (text, annotations, source_path)
    """
    # Parse the XML file
    parser = etree.XMLParser(remove_blank_text=True, remove_comments=True, remove_pis=True)
    tree = etree.parse(xml_file_path, parser)
    root = tree.getroot()
    return convert_tei_root_to_standoff(root)
