def openImages():
    image_list = []
    for file in glob.glob('dataset/iamges/*.jpg'):
        image = Image.open(file)
        image_list.appen(image)
    return image_list

def openPath():
    posesDF = pd.read_csv('dataset/iamges/*.csv')
    print(df.head())
    return posesDF