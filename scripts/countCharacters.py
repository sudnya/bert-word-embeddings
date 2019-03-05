
total = 0
wordCount = 0

for line in open("vocabs/vocab-guttenberg-4k.txt"):
    wordCount += 1
    total += len(line)

print("Average chatacters per word ", (total + 0.0)/wordCount)

