import enchant
d = enchant.Dict("en_US")
d.check("Hello")
print d.suggest("l1ov")