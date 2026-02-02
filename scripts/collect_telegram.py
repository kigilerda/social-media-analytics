import os
import pandas as pd
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

api_id = int(os.getenv("TG_API_ID", "0"))
api_hash = os.getenv("TG_API_HASH", "")
name_chanel = os.getenv("TG_CHANNEL", "")

if not api_id or not api_hash or not name_chanel:
    raise ValueError("Проверь .env: должны быть TG_API_ID, TG_API_HASH, TG_CHANNEL")


async def main(name_chanel, api_id, api_hash):
    # Авторизация клиента и получение сообщений
    async with TelegramClient("session", api_id, api_hash) as client:
        messages = await client.get_messages(name_chanel, limit=5000) 
        return messages


async def run():
    result = await main(name_chanel, api_id, api_hash)

    # создаем DataFrame для сохранения бд
    data = pd.DataFrame(columns=["id_post", "date", "text", "views", "reactions", "comments"])
    i = 0

    # Обрабатываем каждый пост
    for message in result:
        if message.text:
            # Суммарное количество реакций
            if message.reactions:
                count_reactions = sum(
                    [
                        message.reactions.results[r].count
                        for r in range(len(message.reactions.results))
                        if message.reactions.results
                    ]
                )
            else:
                count_reactions = 0

            # Количество комментариев
            if message.replies:
                count_replies = message.replies.replies
            else:
                count_replies = 0

            # Добавляем данные в DataFrame
            data.loc[i] = [message.id, message.date, message.text, message.views, count_reactions, count_replies]
            i += 1

    print(data.head(5))
    print("Rows:", len(data))

    os.makedirs("data/raw", exist_ok=True)
    data.to_csv("data/raw/data_base.csv", index=False, encoding="utf-8")
    print("Saved to data/raw/data_base.csv")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
