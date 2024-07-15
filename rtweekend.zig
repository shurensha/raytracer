const std = @import("std");
const rand = std.crypto.random;

pub const infinity: f32 = std.math.inf(f32);
pub const pi = 3.1415926535897932385;

pub inline fn degrees_to_radians(degrees: f32) f32 {
    return degrees * pi / 180.0;
}

pub inline fn random_double(min: f32, max: f32) f32 {
    return min + (max - min) * rand.float(f32);
}
