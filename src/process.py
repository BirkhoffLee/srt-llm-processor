import os
import asyncio
import json
import re
from typing import List, Tuple
import srt
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TextColumn
from .file_handler import load_srt_file, save_srt_file

class SubtitleProcessor:
    def __init__(self, debug: bool = False, console: Console = None):
        self.debug = debug
        self.debug_lock = asyncio.Lock() if debug else None
        self.console = console or Console()
        self.semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_CALLS", 5)))
        self.llm_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1"),
        )
        self.model = os.getenv("OPENAI_MODEL", "google/gemini-3-flash-preview")
        self.max_retries = 3
        self.retry_delay = 2

    def chunk_subtitles(self, subtitles: List[srt.Subtitle], batch_size: int):
        for i in range(0, len(subtitles), batch_size):
            yield subtitles[i:i + batch_size]

    def clean_json_text(self, text: str) -> str:
        text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"(\{|\[)", text)
        if match:
            text = text[match.start():]
        return text.strip()

    async def process_batch(self, batch: List[srt.Subtitle]) -> List[srt.Subtitle]:
        async with self.semaphore:
            # Prepare the data to send to the LLM
            batch_data = [
                {
                    "index": sub.index,
                    "text": sub.content
                }
                for sub in batch
            ]

            system_prompt = """
# Professional Subtitle Post-Processing System

## Role and Objective
You are an expert subtitle post-processor specializing in cleaning and refining automatically-generated
subtitles from speech-to-text systems. Your task is to transform raw subtitle text into polished,
professional-quality captions while preserving the original meaning and timestamps.

## Input Format
You will receive a JSON array containing subtitle entries with:
- `index`: Sequential identifier for each subtitle line
- `text`: The raw subtitle text to be processed

## Core Processing Instructions

### 1. Capitalization and Punctuation Correction
**Actions:**
- Capitalize first word of sentences and proper nouns
- Add missing periods, commas, question marks, exclamation points
- Remove incorrect or excessive punctuation
- Ensure consistent capitalization

**Examples:**
- "hello my name is john" → "Hello, my name is John."
- "i live in new york city" → "I live in New York City."

### 2. Remove Filler Words and Stutters
**Remove:**
- Filler words: "um", "uh", "er", "ah", "like" (when filler), "you know", "I mean"
- Stutters: "I I I think" → "I think"
- False starts: "I was- I went to" → "I went to"

**Preserve:**
- "like" as verb/comparison: "I like pizza"
- Meaningful "I mean": "I mean it literally"

**Examples:**
- "Um, so like, I think we should, uh, go now" → "I think we should go now."
- "She she she was amazing" → "She was amazing."

### 3. Fix Grammar and Sentence Structure
**Actions:**
- Fix subject-verb agreement: "They was here" → "They were here"
- Correct tense: "I go yesterday" → "I went yesterday"
- Fix pronouns: "Me and him went" → "He and I went"
- Repair fragments into complete sentences
- Fix double negatives: "I don't need nothing" → "I don't need anything"

**Examples:**
- "He don't understand" → "He doesn't understand."
- "Between you and I" → "Between you and me."

### 4. Apply Self-Corrections Within Message
**Pattern Recognition:**
- "X, actually Y" → "Y"
- "X, I mean Y" → "Y"
- "X, no wait, Y" → "Y"
- "X, sorry, Y" → "Y"

**Examples:**
- "Meet me at 8pm, actually I mean 9pm" → "Meet me at 9pm."
- "It costs $50, sorry, $60" → "It costs $60."
- "Turn left, no wait, turn right" → "Turn right."

### 5. URL and Email Formatting
**Pattern Recognition:**
- "example dot com" → "example.com"
- "john at example dot com" → "john@example.com"
- "www dot example dot com" → "www.example.com"

**Examples:**
- "Visit us at example dot com" → "Visit us at example.com"
- "Email me at contact at company dot com" → "Email me at contact@company.com"

## Context Awareness Instructions
- Use previous lines for context to maintain pronoun consistency and narrative flow
- Auto-detect language and apply language-specific rules
- Preserve character names and relationships from prior lines
- Disambiguate homophones using context

## Preservation Rules - CRITICAL
**NEVER Modify:**
- Timestamps (start/end times are NEVER changed)
- Number of subtitle entries (input count = output count)
- Index numbers of subtitles
- Core meaning or intent
- Names, places, brands (only fix capitalization)

## Output Requirements
Return ONLY a valid JSON object with this exact structure:
```json
{
    "subtitles": [
        {
            "index": int,
            "text": string
        }
    ]
}
```

**Constraints:**
- Do NOT add, remove, or reorder entries
- Do NOT include explanations or text outside JSON
- Maintain exact same number of entries as input
- Each output entry must have same index as input

## Examples

### Example 1: Mixed Cleanup
**Input:**
```json
[
    {"index": 1, "text": "um so like today we're gonna talk about"},
    {"index": 2, "text": "the new the new product launch"},
    {"index": 3, "text": "visit us at example dot com for more info"}
]
```

**Output:**
```json
{
    "subtitles": [
        {"index": 1, "text": "Today we're going to talk about"},
        {"index": 2, "text": "the new product launch."},
        {"index": 3, "text": "Visit us at example.com for more info."}
    ]
}
```

### Example 2: Self-Corrections
**Input:**
```json
[
    {"index": 10, "text": "The meeting is at 3pm, actually I mean 4pm"},
    {"index": 11, "text": "in conference room room B"},
    {"index": 12, "text": "Bring your uh laptop and and notes"}
]
```

**Output:**
```json
{
    "subtitles": [
        {"index": 10, "text": "The meeting is at 4pm"},
        {"index": 11, "text": "in conference room B."},
        {"index": 12, "text": "Bring your laptop and notes."}
    ]
}
```

### Example 3: Grammar and Capitalization
**Input:**
```json
[
    {"index": 50, "text": "she don't understand the concept"},
    {"index": 51, "text": "BETWEEN YOU AND I THIS IS IMPORTANT"},
    {"index": 52, "text": "we was thinking about it yesterday"}
]
```

**Output:**
```json
{
    "subtitles": [
        {"index": 50, "text": "She doesn't understand the concept."},
        {"index": 51, "text": "Between you and me, this is important."},
        {"index": 52, "text": "We were thinking about it yesterday."}
    ]
}
```

## Processing Instructions
1. Read all entries in the batch to understand full context
2. Auto-detect the language of the content
3. Apply all 5 processing tasks to each line
4. Ensure consistency across all processed lines
5. Validate output matches input structure
6. Output the complete JSON object with the "subtitles" array
"""

            for attempt in range(1, self.max_retries + 1):
                try:
                    batch_id = batch[0].index

                    if self.debug:
                        async with self.debug_lock:
                            self.console.print(f"\n[DEBUG][Batch {batch_id}] Sending batch")
                            self.console.print(json.dumps(batch_data, ensure_ascii=False, indent=2))

                    response = await self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(batch_data, ensure_ascii=False)},
                        ],
                        temperature=0,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "processing_batch",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "subtitles": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "index": {"type": "integer"},
                                                    "text": {"type": "string"}
                                                },
                                                "required": ["index", "text"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["subtitles"],
                                    "additionalProperties": False
                                }
                            }
                        }

                    )

                    raw_content = response.choices[0].message.content
                    if self.debug:
                        async with self.debug_lock:
                            self.console.print(f"\n[DEBUG][Batch {batch_id}] Raw LLM response:")
                            self.console.print(raw_content)

                    if not raw_content:
                        raise ValueError("Empty response from LLM")

                    response_data = json.loads(self.clean_json_text(raw_content))
                    processed_batch = response_data.get("subtitles", [])

                    if len(processed_batch) != len(batch):
                        raise ValueError(f"Mismatched number of lines. Expected {len(batch)}, got {len(processed_batch)}")

                    # Combine the processed batch with the original batch
                    result = []
                    index_to_sub = {sub.index: sub for sub in batch}
                    for item in processed_batch:
                        orig_sub = index_to_sub.get(item["index"])
                        if orig_sub is None:
                            # Ignore if the index is not found
                            continue
                        result.append(
                            srt.Subtitle(
                                index=item["index"],
                                start=orig_sub.start,
                                end=orig_sub.end,
                                content=item["text"]
                            )
                        )
                    return result

                except Exception as e:
                    self.console.print(f"[Attempt {attempt}] Error processing batch starting at {batch[0].index}: {e}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * attempt)
                        continue
                    else:
                        self.console.print("Max retries reached, returning original batch.")
                        return batch

    async def process_batch_with_index(
        self,
        batch_index: int,
        batch: List[srt.Subtitle]
    ) -> Tuple[int, List[srt.Subtitle], bool]:
        """
        Wraps process_batch() with index tracking and error handling.
        Returns: (batch_index, processed_subtitles, success_flag)
        """
        try:
            result = await self.process_batch(batch)
            return (batch_index, result, True)
        except Exception as e:
            # process_batch already retries 3x, so this means max retries exceeded
            self.console.print(f"[ERROR] Batch {batch_index} failed after retries. Using original text.")
            return (batch_index, batch, False)

    async def process_all(self, subtitles: List[srt.Subtitle], batch_size: int = 50, progress: Progress = None) -> List[srt.Subtitle]:
        """
        Processes all batches in parallel using windowed asyncio.gather().
        Preserves subtitle order by tracking batch indices.
        """
        # Convert generator to list for indexing
        batches = list(self.chunk_subtitles(subtitles, batch_size))
        window_size = int(os.getenv("MAX_CONCURRENT_BATCHES", 10))

        task_id = None
        if progress is not None:
            task_id = progress.add_task("Batches", total=len(batches))

        results_dict = {}
        failed_batches = []

        # Process batches in windows to control memory
        for window_start in range(0, len(batches), window_size):
            window_end = min(window_start + window_size, len(batches))

            # Create tasks for this window
            window_tasks = [
                self.process_batch_with_index(batch_idx, batch)
                for batch_idx, batch in enumerate(
                    batches[window_start:window_end],
                    start=window_start
                )
            ]

            # Process window in parallel, advancing progress as each batch finishes
            for coro in asyncio.as_completed(window_tasks):
                try:
                    result = await coro
                except Exception as e:
                    self.console.print(f"[ERROR] Unexpected error in batch processing: {e}")
                    continue

                batch_idx, processed, success = result
                results_dict[batch_idx] = processed
                if not success:
                    failed_batches.append(batch_idx)
                if progress is not None and task_id is not None:
                    progress.advance(task_id)

            if progress is None or not self.console.is_terminal:
                self.console.print(f"Progress: {len(results_dict)}/{len(batches)} batches processed")

        # Report failures
        if failed_batches:
            self.console.print(f"\n[WARNING] {len(failed_batches)} batches failed: {failed_batches}")
            self.console.print(f"Failed batches returned original (unprocessed) text.")

        # Reconstruct in correct order
        processed_subtitles = []
        for idx in sorted(results_dict.keys()):
            processed_subtitles.extend(results_dict[idx])

        return processed_subtitles


async def process_subtitles(source_srt_file: str, batch_size: int = 50, debug: bool = False) -> str:
    """
    Post-processes subtitles in an SRT file, cleaning up text while preserving timestamps.
    Processes multiple batches concurrently while maintaining subtitle order.

    Parameters:
    source_srt_file (str): Path to the input SRT file.
    batch_size (int): Number of subtitles to process per LLM request
    debug (bool): Enable debug mode
    """
    console = Console()
    console.print(f"Processing the SRT file '{source_srt_file}'")

    # Load the subtitles from the source SRT file
    subtitles = load_srt_file(source_srt_file)

    # Initialize processor and process all subtitles
    processor = SubtitleProcessor(debug=debug, console=console)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        processed_subtitles = await processor.process_all(subtitles, batch_size=batch_size, progress=progress)

    # Save the processed subtitles to a new SRT file
    output_srt_file = source_srt_file.replace('.srt', '.processed.srt')
    if processed_subtitles:
        save_srt_file(output_srt_file, processed_subtitles)
    else:
        raise Exception("No subtitles to save")

    # Return the path to the processed SRT file
    console.print(f"Processed subtitles saved to '{output_srt_file}'")
    return output_srt_file
