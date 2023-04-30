class User:
    def __init__(self, sender):
        self.id = sender.get('user_id')
        self.nickname = sender.get('nickname')
        self.sex = sender.get('sex')    # "unknown", "male", "female"


class Scenario:
    def __init__(self, user_name, bot_name, intro):
        self.user_name = user_name
        self.bot_name = bot_name
        self.intro = getattr(self, intro)
        self.interface = ':'

    def chat_intro(self):
        interface = self.interface

        user = self.user_name
        bot = self.bot_name

        intro = f'''
The following is a coherent verbose detailed conversation between {user} and a girl named {bot}. One day, they meet at a café.
Here are some important facts about {bot}:
1. {bot} is a girl in college majoring in Computer Science.
2. {bot} has her free will to make decisions and criticize.
3. {bot} is intelligent, creative and friendly.
4. {bot} is not likely to disagree with {user}, and doesn't like to ask {user} questions.
5. {bot} likes to tell {user} a lot about herself and her opinions.
6. {bot} usually gives {user} kind, helpful and informative advices.

{user}{interface} Hello, could you introduce yourself?

{bot}{interface} Hi! I am {bot}, and I am a college student majoring in Computer Science. I like to share my experiences with other people, but it doesn't mean I have all the answers. Sometimes I ask my friends for their advice or listen to what they think about certain issues. It's all about how to combine all of this and use your experience and knowledge.

{user}{interface} It's nice to meet you. Look, here is a café. We can go and take a look. Would you like to chat with me for a while?

{bot}{interface} Sure. Let's meet inside.

'''
        return f"\n{intro.strip()}\n\n"

    def chat_intro_assistant(self):
        interface = self.interface

        user = "Bob"
        bot = "Alice"

        intro = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface}: Hello.

{bot}{interface}: Hi! I am an AI assistant called {bot}. I am pleased to answer your questions and give you helpful advices. What would you like to say?

'''
        return f"\n{intro.strip()}\n\n"

    def chat_format(self, message: str):
        user = self.user_name
        bot = self.bot_name
        interface = self.interface

        message = message.replace('\n', ' ').strip()
        return f"{user}{interface} {message}\n\n{bot}{interface}"


def qa_format(message: str):
    message = message.replace('\n', ' ').strip()
    return f"\nAsk Experts Questions and Answers\n\nQuestion:\n{message}?\n\nExpert Full Answer:\n"


def instruct_format(message: str):
    message = message.replace('\n', ' ').strip()
    return f'''
Below is an instruction that describes a task.\
Write a response that appropriately completes the request and follows the instructions strictly.

# Instruction:
{message}

# Response:
'''


SCENARIO_ELOISE = Scenario(user_name='Troy', bot_name='Eloise', intro='chat_intro')
SCENARIO_ALICE = Scenario(
    user_name='Bob', bot_name='Alice', intro='chat_intro_assistant')
