#!/usr/bin/env python3
"""
A simple async load test script for the /generate endpoint.
Usage examples:
    python load_test.py --url http://localhost:8000/generate --concurrency 4 --requests 20
    python load_test.py --url http://localhost:8000/generate --concurrency 8 --requests 100 --text "hello"
    python load_test.py --url http://localhost:8000/generate --concurrency 2 --requests 10 --image test.jpg

This uses `httpx` for async HTTP and reports successes, failures and latencies.
"""

import argparse
import asyncio
import time
from statistics import mean

import httpx
# no longer need io import for reused bytes


async def worker(name, client, url, queue, results, text, image_bytes, semaphore):
    while True:
        i = await queue.get()
        if i is None:
            queue.task_done()
            break
        payload = {"text": text}
        files = None
        if image_bytes:
            # reuse bytes in memory for every request instead of opening file each time
            # pass bytes as the content of multipart upload
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}

        await semaphore.acquire()
        start = time.perf_counter()
        try:
            r = await client.post(url, data=payload, files=files, timeout=60)
            elapsed = time.perf_counter() - start
            results.append((r.status_code, elapsed))
            print(f"[{name}] req {i} -> {r.status_code} in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append((None, elapsed))
            print(f"[{name}] req {i} failed: {e} (took {elapsed:.2f}s)")
        finally:
            # nothing to close for reused bytes
            semaphore.release()
            queue.task_done()


async def run(url, concurrency, total_requests, text, image_path, rate):
    queue = asyncio.Queue()
    # producer with optional rate limiting
    async def producer():
        if rate and rate > 0:
            interval = 1.0 / rate
            for i in range(total_requests):
                await queue.put(i)
                await asyncio.sleep(interval)
        else:
            for i in range(total_requests):
                await queue.put(i)
        # put sentinel None for each worker to shut them down
        for _ in range(concurrency):
            await queue.put(None)

    results = []
    semaphore = asyncio.Semaphore(concurrency)  # limit to concurrency in parallel

    # Configure connection limits to avoid creating huge numbers of sockets
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    # Read the image into memory once (avoid repeated disk I/O)
    image_bytes = None
    if image_path:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

    async with httpx.AsyncClient(limits=limits) as client:
        # create workers
        workers = [asyncio.create_task(worker(f"W{n}", client, url, queue, results, text, image_bytes, semaphore)) for n in range(concurrency)]

        # start producer
        prod = asyncio.create_task(producer())

        await prod
        await queue.join()

        for w in workers:
            w.cancel()

    # report
    successes = [r for r in results if r[0] == 200]
    failures = [r for r in results if r[0] != 200]
    latencies = [r[1] for r in results if r[0] is not None]

    print("\n=== RESULTS ===")
    print(f"Total requests: {len(results)}")
    print(f"Successes (200): {len(successes)}")
    print(f"Failures: {len(failures)}")
    if latencies:
        print(f"Min latency: {min(latencies):.2f}s")
        print(f"Max latency: {max(latencies):.2f}s")
        print(f"Avg latency: {mean(latencies):.2f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/generate", help="Target endpoint")
    p.add_argument("--concurrency", type=int, default=2, help="Number of parallel workers")
    p.add_argument("--requests", type=int, default=10, help="Total requests to send")
    p.add_argument("--text", default="test", help="Text content to send")
    p.add_argument("--image", default=None, help="Optional image file path to send")
    p.add_argument("--rate", type=float, default=0, help="Requests per second rate (0 = as fast as possible)")
    args = p.parse_args()

    asyncio.run(run(args.url, args.concurrency, args.requests, args.text, args.image, args.rate))


if __name__ == "__main__":
    main()
