import os
import sys
import imp
import traceback


os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
REPO_DIR = os.getcwd()
TOOLS_DIR = os.path.join(REPO_DIR, 'tools')
TESTS_DIR = os.path.join(REPO_DIR, 'tests')

things_to_run = [
    (TESTS_DIR, 'test_dragon_diagram'),
    (TOOLS_DIR, 'dragon_diagram')
]

for thing in things_to_run:
    try:
        PATH = sys.path[:]
        sys.path.insert(0, thing[0])
        module = __import__(thing[1])
        imp.reload(module)
    except Exception:
        traceback.print_exc()
    finally:
        sys.path = PATH

    filename = os.path.join(*thing) + '.py'
    try:
        result = exec(compile(open(filename).read(), filename, 'exec'))
        print(f"result for thing {thing}: {repr(result)}")
    except Exception:
        traceback.print_exc()
        break
