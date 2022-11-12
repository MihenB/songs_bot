from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from songAkinator import akinate, learn
from config import TOKEN


bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
noise, lemmatizer, vectorizer, clf = learn()


@dp.message_handler(commands="start")
async def cmd_start(message: types.Message):
    await message.reply("Привет!\n"
                        "Я \"умею\" угадывать жанр песни по тексту* \n\n"
                        "*только если очень хорошо обучусь)))))")
    await message.reply("Напиши мне текст песни и я возможно угадаю ее жанр (blues, country, hip hop, "
                        "jazz, pop, reggae, rock) :)")


@dp.message_handler(commands="help")
async def process_help_command(message: types.Message):
    await message.reply("Просто напиши текст песни, смирись и не дудось)")


@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, "Надо подумать...")
    text = msg.text
    print(msg.from_user.id)
    if len(text) < 50:
        result = 'Ты меня за дурака не держи!))'
        await bot.send_message(msg.from_user.id, str(result))
    else:
        result = akinate(noise, lemmatizer, vectorizer, clf, text)
        await bot.send_message(msg.from_user.id, "Возможно это: " + str(result))


if __name__ == '__main__':
    executor.start_polling(dp)
