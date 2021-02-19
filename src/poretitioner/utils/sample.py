a = [6, 7, 8, 9]

# black wants no space before : but also wants space before :
ix = (2, 3)
b = a[ix[0] : ix[1]]

# black is happy
i, j = (2, 3)
b = a[i:j]
