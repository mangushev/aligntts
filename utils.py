
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ','\"','\'',';','.',',','-','(',')',':','!','?']

char_to_ix = { ch:i for i,ch in enumerate(alphabet) } 
ix_to_char = { i:ch for i,ch in enumerate(alphabet) }

nrof_labels = len(alphabet)
nrof_classes = nrof_labels
