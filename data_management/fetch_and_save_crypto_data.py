import asyncio
import os
import ccxt.async_support as ccxt
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

base_dir = "data/stock/csv"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

timeframes = ["5m", "15m", "1h", "4h", "1d", "1w"]
concurrency_limit = 20  # Adjust based on your needs and rate limits


async def fetch_and_save(symbol, timeframe, exchange, semaphore: asyncio.Semaphore):
    async with semaphore:  # This will limit the number of concurrent tasks
        filename = f"{base_dir}/{symbol.replace('/', '_')}_{timeframe}.csv"
        if os.path.exists(filename):
            logging.info(
                f"Data for {symbol} on timeframe {timeframe} already exists. Skipping download."
            )
            return

        since = exchange.parse8601("2017-01-01T00:00:00Z")
        all_ohlcv = []
        retry_count = 0
        max_retries = 5  # Maximum number of retries for fetching data

        while True:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                retry_count = 0  # Reset retry count on successful fetch
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count > max_retries:
                    logging.error(
                        f"Max retries exceeded for {symbol} on timeframe {timeframe}."
                    )
                    return
                logging.warning(
                    f"Timeout error for {symbol} on timeframe {timeframe}. Retrying... Attempt {retry_count}"
                )
                await asyncio.sleep(2**retry_count)  # Exponential backoff

        if all_ohlcv:
            df = pd.DataFrame(
                all_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df.to_csv(filename, index=False)
            logging.info(
                f"Completed fetching {len(all_ohlcv)} entries for {symbol} on timeframe {timeframe}. Data saved to {filename}"
            )


async def main():
    semaphore = asyncio.Semaphore(concurrency_limit)
    exchange = ccxt.binance({"enableRateLimit": True})
    symbols = list((await exchange.load_markets()).keys())

    tasks = [
        fetch_and_save(symbol, timeframe, exchange, semaphore)
        for symbol in symbols
        for timeframe in timeframes
    ]
    await asyncio.gather(*tasks)
    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
