import aiml,os

kernel=aiml.Kernel()

if os.path.isfile("bot_mind.brn"):
    kernel.bootstrap(brainFile="bot_mind.brn")
else:
    kernel.bootstrap(learnFiles=os.path.abspath("aiml/std-startup.xml"),commands="load aiml b")
    kernel.saveBrain("bot_mind.brn")

while True:
    msg=input('Enter message to bot')
    if msg=="quit":
        exit()
    elif msg=="save":
        kernel.saveBrain("bot_mind.brn")
    else:
        bot_res=kernel.respond(msg)
        print(bot_res)