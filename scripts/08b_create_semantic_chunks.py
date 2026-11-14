
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "merged_pages.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "chunks_v2.jsonl"
REPORT_FILE = ROOT / "data" / "processed" / "chunk_quality_report_v2.json"

TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TARGET_TOKENS = 450
MIN_TOKENS = 300
MAX_TOKENS = 600

# Used only when a single sentence/unit exceeds MAX_TOKENS.
LONG_UNIT_SIZE = 500


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    trust_remote_code=True
)


def count_tokens(text):
    """Count tokens without adding special tokens."""
    if not text.strip():
        return 0

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


# --------------------------------------------------
# INPUT
# --------------------------------------------------

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                )

    return records


# --------------------------------------------------
# SEMANTIC UNIT EXTRACTION
# --------------------------------------------------

def split_paragraph_into_sentences(paragraph):
    """
    Split a paragraph at likely sentence boundaries.

    Original sentence text is retained. This is a lightweight
    boundary detector, not a linguistic parser.
    """

    paragraph = paragraph.strip()

    if not paragraph:
        return []

    # Split after sentence-ending punctuation when followed
    # by whitespace and a likely sentence/heading start.
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'•\-])",
        paragraph
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def extract_units(page):
    """
    Convert one page into ordered text units.

    Each unit retains its original page number.
    """

    content = page.get("content", "").strip()

    if not content:
        return []

    # Separate clearly distinct paragraphs.
    paragraphs = re.split(r"\n\s*\n+", content)

    units = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        sentences = split_paragraph_into_sentences(paragraph)

        for sentence in sentences:
            units.append({
                "text": sentence,
                "page_number": page["page_number"]
            })

    return units


# --------------------------------------------------
# SPLIT EXCEPTIONALLY LONG UNITS
# --------------------------------------------------

def split_long_unit(unit):
    """
    Split a single oversized sentence/unit into token-sized
    pieces. This fallback is used only when a unit exceeds
    MAX_TOKENS.
    """

    text = unit["text"]
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )

    offsets = tokenized["offset_mapping"]

    if len(offsets) <= MAX_TOKENS:
        return [unit]

    pieces = []

    for start in range(0, len(offsets), LONG_UNIT_SIZE):
        end = min(start + LONG_UNIT_SIZE, len(offsets))

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        piece_text = text[char_start:char_end].strip()

        if piece_text:
            pieces.append({
                "text": piece_text,
                "page_number": unit["page_number"],
                "was_token_split": True
            })

    return pieces


def prepare_units(pages):
    units = []

    for page in pages:
        page_units = extract_units(page)

        for unit in page_units:
            if count_tokens(unit["text"]) > MAX_TOKENS:
                units.extend(split_long_unit(unit))
            else:
                units.append(unit)

    return units


# --------------------------------------------------
# CHUNK PACKING
# --------------------------------------------------

def chunk_token_count(units):
    return count_tokens("\n\n".join(
        unit["text"] for unit in units
    ))



def pack_units(units):
    """
    Pack sentence-level units into chunks without changing
    their original order.

    Hard maximum: 600 tokens.
    Target: 450 tokens.
    Short final chunks are retained and flagged for review.

    No artificial overlap or backward movement of units.
    """

    chunks = []
    current = []

    for unit in units:

        unit_tokens = count_tokens(unit["text"])

        # A single oversized unit should already have been
        # split by prepare_units(). Fail explicitly if not.
        if unit_tokens > MAX_TOKENS:
            raise ValueError(
                f"Unit exceeds maximum: {unit_tokens} tokens. "
                f"Page: {unit['page_number']}"
            )

        candidate = current + [unit]
        candidate_tokens = chunk_token_count(candidate)

        # Never allow a chunk to exceed the maximum.
        if current and candidate_tokens > MAX_TOKENS:
            chunks.append(current)
            current = [unit]
            continue

        current.append(unit)

        # Flush at or above the target.
        if chunk_token_count(current) >= TARGET_TOKENS:
            chunks.append(current)
            current = []

    # Retain the final chunk without moving units backwards.
    # Retain the final chunk, or merge it with the previous
    # chunk when safe within this same contiguous segment.
    if current:

        if chunks:
            combined = chunks[-1] + current
            combined_tokens = chunk_token_count(combined)

            if combined_tokens <= MAX_TOKENS:
                chunks[-1] = combined
            else:
                chunks.append(current)

        else:
            chunks.append(current)

    return chunks

# --------------------------------------------------
# PAGE CONTINUITY
# --------------------------------------------------

def split_into_contiguous_segments(pages):
    """
    Create separate segments whenever source page numbers
    are not consecutive.

    This prevents chunks from silently crossing missing pages.
    """

    if not pages:
        return []

    segments = []
    current_segment = [pages[0]]

    for page in pages[1:]:
        previous_page = current_segment[-1]["page_number"]

        if page["page_number"] == previous_page + 1:
            current_segment.append(page)
        else:
            segments.append(current_segment)
            current_segment = [page]

    if current_segment:
        segments.append(current_segment)

    return segments


# --------------------------------------------------
# BUILD CHUNK RECORDS
# --------------------------------------------------

def build_chunk_record(
    doc_id,
    source_file,
    chunk_index,
    units
):
    content = "\n\n".join(
        unit["text"] for unit in units
    ).strip()

    source_pages = sorted(set(
        unit["page_number"] for unit in units
    ))

    token_count = count_tokens(content)

    return {
        "chunk_id": f"{doc_id}_chunk_{chunk_index:04d}",
        "doc_id": doc_id,
        "source_file": source_file,
        "chunk_index": chunk_index,
        "content": content,
        "token_count": token_count,
        "source_pages": source_pages,
        "page_start": min(source_pages),
        "page_end": max(source_pages),
        "is_underfilled": token_count < MIN_TOKENS,
        "is_overfilled": token_count > MAX_TOKENS,
        "contains_token_split": any(
            unit.get("was_token_split", False)
            for unit in units
        )
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"Reading: {INPUT_FILE}")

    records = load_jsonl(INPUT_FILE)

    # Only usable pages are eligible for chunking.
    usable_pages = [
        record for record in records
        if record.get("is_usable") is True
        and record.get("content", "").strip()
    ]

    documents = defaultdict(list)

    for page in usable_pages:
        documents[page["doc_id"]].append(page)

    all_chunks = []
    document_segment_counts = {}
    document_chunk_counts = {}

    for doc_id in sorted(documents):

        pages = sorted(
            documents[doc_id],
            key=lambda item: item["page_number"]
        )

        source_file = pages[0].get("source_file", "")

        segments = split_into_contiguous_segments(pages)

        document_segment_counts[doc_id] = len(segments)
        document_chunk_counts[doc_id] = 0

        chunk_index = 1

        for segment in segments:

            units = prepare_units(segment)

            if not units:
                continue

            packed_chunks = pack_units(units)

                # Validate that chunk packing preserves source order.
            previous_page = -1

            for packed_units in packed_chunks:
                unit_pages = [
                    unit["page_number"]
                    for unit in packed_units
                ]

                if unit_pages != sorted(unit_pages):
                    raise ValueError(
                        f"Source-page order violation in {doc_id}: {unit_pages}"
                    )

                if unit_pages and unit_pages[0] < previous_page:
                    raise ValueError(
                        f"Chunk order violation in {doc_id}: "
                        f"previous page ended at {previous_page}, "
                        f"next chunk starts at {unit_pages[0]}"
                    )

                previous_page = unit_pages[-1]

            for packed_units in packed_chunks:

                chunk = build_chunk_record(
                    doc_id=doc_id,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    units=packed_units
                )

                all_chunks.append(chunk)
                chunk_index += 1

        document_chunk_counts[doc_id] = chunk_index - 1

    # --------------------------------------------------
    # WRITE CHUNKS
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    # --------------------------------------------------
    # QUALITY REPORT
    # --------------------------------------------------

    token_counts = [
        chunk["token_count"] for chunk in all_chunks
    ]

    status_counts = Counter()

    for chunk in all_chunks:
        if chunk["is_underfilled"]:
            status_counts["underfilled"] += 1
        elif chunk["is_overfilled"]:
            status_counts["overfilled"] += 1
        else:
            status_counts["within_range"] += 1

    report = {
        "tokenizer": TOKENIZER_NAME,
        "target_tokens": TARGET_TOKENS,
        "minimum_tokens": MIN_TOKENS,
        "maximum_tokens": MAX_TOKENS,
        "overlap_tokens": 0,
        "total_documents": len(documents),
        "total_usable_pages": len(usable_pages),
        "total_segments": sum(document_segment_counts.values()),
        "total_chunks": len(all_chunks),
        "total_chunk_tokens": sum(token_counts),
        "average_chunk_tokens": (
            round(sum(token_counts) / len(token_counts), 2)
            if token_counts else 0
        ),
        "minimum_actual_tokens": min(token_counts) if token_counts else 0,
        "maximum_actual_tokens": max(token_counts) if token_counts else 0,
        "chunk_status_counts": dict(status_counts),
        "chunks_per_document": document_chunk_counts,
        "segments_per_document": document_segment_counts,
        "token_split_chunks": sum(
            chunk["contains_token_split"]
            for chunk in all_chunks
        ),
        "underfilled_chunk_ids": [
            chunk["chunk_id"]
            for chunk in all_chunks
            if chunk["is_underfilled"]
        ],
        "overfilled_chunk_ids": [
            chunk["chunk_id"]
            for chunk in all_chunks
            if chunk["is_overfilled"]
        ]
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # --------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("SEMANTIC CHUNKING COMPLETE")
    print("=" * 55)

    print(f"Usable pages: {len(usable_pages)}")
    print(f"Documents: {len(documents)}")
    print(f"Contiguous segments: {report['total_segments']}")
    print(f"Total chunks: {report['total_chunks']}")
    print(f"Total chunk tokens: {report['total_chunk_tokens']}")
    print(f"Average chunk tokens: {report['average_chunk_tokens']}")
    print(f"Minimum chunk tokens: {report['minimum_actual_tokens']}")
    print(f"Maximum chunk tokens: {report['maximum_actual_tokens']}")

    print("\nCHUNK STATUS:")
    for status, count in status_counts.items():
        print(f"{status}: {count}")

    print("\nCHUNKS PER DOCUMENT:")
    for doc_id, count in document_chunk_counts.items():
        print(f"{doc_id}: {count}")

    print("\nSEGMENTS PER DOCUMENT:")
    for doc_id, count in document_segment_counts.items():
        print(f"{doc_id}: {count}")

    print("\nOUTPUT FILES:")
    print(OUTPUT_FILE)
    print(REPORT_FILE)

    if report["overfilled_chunk_ids"]:
        print("\nWARNING: Overfilled chunks detected.")

    if report["underfilled_chunk_ids"]:
        print("\nWARNING: Underfilled chunks detected.")
        print("These need inspection; short isolated segments may be unavoidable.")

    print("\nChunk generation finished.")


if __name__ == "__main__":
    main()