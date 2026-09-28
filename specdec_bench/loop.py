# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from tqdm.asyncio import tqdm
from .utils import encode_chat, decode_chat


async def tqdm_gather(*fs, return_exceptions=False, **kwargs):
    if not return_exceptions:
        return await tqdm.gather(*fs, **kwargs)

    async def wrap(f):
        try:
            return await f
        except Exception as e:
            return e

    return await tqdm.gather(*map(wrap, fs), **kwargs)

async def run_loop(
    runner,
    dataset,
    tokenizer,
    output_length,
    postprocess,
    concurrency=10,
    end_id=-1,
    show_progress=False,
    completions=False,
    chat_template_args={},
):
    """
    Async version of run_loop with concurrency control using a semaphore.

    Args:
        runner: The model runner instance
        dataset: The dataset containing requests
        tokenizer: The tokenizer instance
        output_length: Maximum output length
        concurrency: Maximum number of concurrent requests (default: 10)
    """
    semaphore = asyncio.Semaphore(concurrency)
    max_length = output_length

    async def process_single_request(request, i):
        """Process a single request with all its conversation turns."""
        async with semaphore:
            messages = []
            if request.system_prompt is not None:
                messages.append({"role": "system", "content": request.system_prompt})

            for turn_id, question in enumerate(request.turns):
                messages.append({"role": "user", "content": question})
                entry_encoded = encode_chat(
                    tokenizer,
                    messages,
                    chat_template_args=chat_template_args,
                    completions=completions,
                )

                # Run the async runner.run directly
                output_tokens = await runner.run(
                    entry_encoded, max_length, end_id, request_id=i, turn_id=turn_id
                )
                output_text = decode_chat(tokenizer, output_tokens["output_ids"][0])
                output_text = postprocess(output_text)
                messages.append({"role": "assistant", "content": output_text})

            return messages

    tasks = [process_single_request(request, i) for i, request in enumerate(dataset.data)]
    if show_progress:
        text_outputs = await tqdm_gather(
            *tasks,
            return_exceptions=True,
            desc=f"Running requests (concurrency={concurrency})",
        )
    else:
        text_outputs = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions and handle them
    for i, result in enumerate(text_outputs):
        if isinstance(result, Exception):
            print(f"Error processing request {i}/{dataset.data[i].question_id}: {result}")
            raise result

    runner.process_metrics_final(text_outputs)
    return text_outputs
