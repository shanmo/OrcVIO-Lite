import csv
def main(incsv, outtxt, skipfirst=1):
    with open(incsv) as infile:
        inreader = csv.reader(infile)
        with open(outtxt, 'w') as outfile:
            outwriter = csv.writer(
                outfile, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
            for linenum, row in enumerate(inreader):
                if linenum >= skipfirst:
                    row[0] = int(row[0]) / 1e9
                    outwriter.writerow(row)

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])

