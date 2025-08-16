def Find_URLS(string):
    x=string.split()
    res=[]
    for i in x:
        if i.startswith("https:") or i.startswith("http:"):
            i=i[:len(i)-4]
            res.append(i)
    return res