# import json

# with open('data/rvos/meta_expressions/train/meta_expressions.json', 'r') as f:
#     content = json.load(f)
#     content = content['videos']

# vocab = set() 
# count = 0 
# for filename, expressions in content.items(): 
#     count += 1
#     for index, exp in expressions['expressions'].items():
#         exp = exp['exp'].split()
#         for word in exp:
#             vocab.add(word.strip())

# with open('data/rvos/vocab', 'w') as f:
#     for word in vocab:
#         f.write(word + '\n')

a = 'asnkdfkla,'
print(a.split(','))