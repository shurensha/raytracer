const std = @import("std");

pub const infinity: f32 = std.math.inf(f32);
pub const pi = 3.1415926535897932385;

pub inline fn degrees_to_radians(degrees: f32) f32 {
    return degrees * pi / 180.0;
}
