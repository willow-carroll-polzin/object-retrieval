def openImages():
    image_list = []
    for file in glob.glob('/dataset/images/*.jpg'):
        image = Image.open(file)
        image_list.appen(image)
    return image_list

def openPath():
    posesDF = pd.read_csv('/dataset/poses/*.csv')
    print(df.head())
    return posesDF