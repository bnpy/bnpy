import os
import nose

testroot = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
if not testroot.endswith(os.path.sep):
    testroot = testroot + os.path.sep

CMD = "%s -v --nocapture --attr=fast" % (testroot)
print(CMD)
result = nose.run(argv=CMD.split())
