../build/generate_aruco_markers ../markers

convert ../markers/*.png ../markers/markers.pdf

# evince ../markers/markers.pdf

pdfnup ../markers/markers.pdf --nup 5x5 --delta "1cm 1cm" --noautoscale true --paper letterpaper --outfile ../markers/multiple.pdf

evince ../markers/multiple.pdf
