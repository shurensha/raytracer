all: 
	zig run main.zig > image.ppm && imv image.ppm
