#create empty file to be writen to
file = open("annotation-id.geojson", "w")
count = 0

#read original file
with open('/Users/philipp/Projects/PycharmProjects/las-exploration/annotations.json', 'r')as myfile:
    for line in myfile:

       #lines that don't need to be edited with an 'id'
       if not line.startswith('      "type": '):
            file.write(line)
       else:
            #lines that need to be edited
            count = count +1
            idNr = str(count)
            file.write(line[0:25] + '"id":'+ '"'+ idNr + '",')

file.close()