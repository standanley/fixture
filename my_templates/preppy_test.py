import preppy


#mymodule = preppy.getModule('amp_test.prep')
mymodule = preppy.getModule('', sourcetext='''
{{def()}}
Hello
{{2+2}}
''')
ans = mymodule.get()
print(ans)

