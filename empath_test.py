from empath import Empath
lexicon = Empath()

lexicon.analyze("he hit the other person", normalize=True)
lexicon.analyze("he hit the other person", categories=["violence"])
lexicon.analyze("he hit the other person", categories=["violence"], normalize=True)
lexicon.create_category("colors",["red","blue","green"])
lexicon.analyze("My favorite color is blue", categories=["colors"], normalize=True)
lexicon.create_category("cold_war", ["cold_war"], model="nytimes")
lexicon.create_category("cold_war", ["cold_war"], model="nytimes", size=300)
