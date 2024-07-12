const std = @import("std");
const math = std.math;

const vec3 = @Vector(3, f32);
const point3 = vec3;
const color = @Vector(3, f32);

fn dot(u: vec3, v: vec3) f32 {
    return @reduce(.Add, u * v);
}

fn vec3_splat(scalar: f32) vec3 {
    return @as(vec3, @splat(scalar));
}

fn length(u: vec3) f32 {
    return math.sqrt(length_squared(u));
}

fn unit_vector(u: vec3) vec3 {
    return u / vec3_splat(length(u));
}

fn length_squared(u: vec3) f32 {
    return u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
}

fn write_color(out: anytype, c: color) !void {
    const r = c[0];
    const g = c[1];
    const b = c[2];

    const rbyte = std.math.lossyCast(u8, 255.999 * r);
    const gbyte = std.math.lossyCast(u8, 255.999 * g);
    const bbyte = std.math.lossyCast(u8, 255.999 * b);

    try out.print("{} {} {}\n", .{ rbyte, gbyte, bbyte });
}

const Ray = struct {
    orig: point3,
    dir: vec3,
    pub fn init(orig: point3, dir: vec3) Ray {
        return Ray{
            .orig = orig,
            .dir = dir,
        };
    }

    pub fn origin(self: *const Ray) point3 {
        return self.orig;
    }

    pub fn direction(self: *const Ray) vec3 {
        return self.dir;
    }

    pub fn at(self: *const Ray, t: f32) point3 {
        return self.orig + vec3_splat(t) * self.dir;
    }

    pub fn ray_color(self: *const Ray) color {
        const t = hit_sphere(point3{ 0, 0, -1.0 }, 0.5, self.*);
        if (t > 0.0) {
            const N = unit_vector(self.*.at(t) - vec3{ 0, 0, -1 });
            return vec3_splat(0.5) * color{ N[0] + 1.0, N[1] + 1.0, N[2] + 1.0 };
        }

        const unit_direction = unit_vector(self.direction());
        // map -1 to 1 to 0 to 1
        const a = 0.5 * (unit_direction[1] + 1.0);

        return vec3_splat(1.0 - a) * color{ 1.0, 1.0, 1.0 } + vec3_splat(a) * color{ 0.5, 0.7, 1.0 };
    }
};

// find t of R = Q + td, which makes R intersect the sphere
fn hit_sphere(center: point3, radius: f32, r: Ray) f32 {
    const oc = center - r.origin();
    const a = dot(r.direction(), r.direction());
    const h = dot(r.direction(), oc);
    const c = length_squared(oc) - radius * radius;
    const discriminant = h * h - a * c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (h - math.sqrt(discriminant)) / a;
    }
}

pub fn main() !void {
    const aspect_ratio: f32 = 16.0 / 9.0;
    const image_width: u32 = 400;
    const image_height = @max(1, @as(u32, @intFromFloat(@floor(image_width /
        aspect_ratio))));

    // camera
    const focal_length: f32 = 1.0;
    const viewport_height: f32 = 2.0;
    const viewport_width = viewport_height * image_width / image_height;
    const camera_center = point3{ 0, 0, 0 };

    const viewport_u = vec3{ viewport_width, 0, 0 };
    const viewport_v = vec3{ 0, -viewport_height, 0 };

    const pixel_delta_u = viewport_u / vec3_splat(image_width);
    const pixel_delta_v = viewport_v / vec3_splat(image_height);

    const viewport_upper_left = camera_center - vec3{ 0, 0, focal_length } - viewport_u / vec3_splat(2) - viewport_v / vec3_splat(2);
    const pixel00_loc = viewport_upper_left + vec3_splat(0.5) * (pixel_delta_u + pixel_delta_v);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("P3\n", .{});
    try stdout.print("{} {}\n", .{ image_width, image_height });
    try stdout.print("255\n", .{});

    for (0..image_height) |j| {
        std.log.info("Scanlines remaining: {}", .{(image_height - j)});
        for (0..image_width) |i| {
            const pixel_center = pixel00_loc +
                (vec3_splat(@floatFromInt(i)) * pixel_delta_u) + (vec3_splat(@floatFromInt(j)) * pixel_delta_v);
            const ray_direction = pixel_center - camera_center;
            const r = Ray.init(camera_center, ray_direction);

            const pixel_color = r.ray_color();

            try write_color(&stdout, pixel_color);
        }
    }

    std.log.info("focal length: {}", .{focal_length});
    std.log.info("image width: {}", .{image_width});
    std.log.info("image height: {}", .{image_height});
    std.log.info("viewport width: {}", .{viewport_width});
    std.log.info("viewport height: {}", .{viewport_height});
    std.log.info("camera: ({},{},{})", .{ camera_center[0], camera_center[1], camera_center[2] });

    std.log.info("Done\n", .{});
}
