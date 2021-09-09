
import sys

if __name__ == "__main__":

    fileName = sys.argv[1]

    with open(fileName, 'rt') as inHandle:
        for currLine in inHandle:
            if not currLine.startswith('#'):
                sys.stdout.write(currLine)

