import sys
import pymupdf

pdf = pymupdf.open("/mnt/c/Users/ivanniko/OneDrive - Hereon/Papers/paper-dealii-agent/main.pdf")
out = "/home/vladimir/work/dealii-agent/paper-bench/pages"
import os
os.makedirs(out, exist_ok=True)
for n in [int(a) for a in sys.argv[1:]]:
    pdf[n - 1].get_pixmap(dpi=70).save(f"{out}/p{n}.png")
print(len(pdf), "pages")
