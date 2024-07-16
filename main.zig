const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const math = std.math;
const assert = std.debug.assert;
const rtweekend = @import("./rtweekend.zig");

const Interval = struct {
    min: f32 = math.inf(f32),
    max: f32 = -math.inf(f32),

    pub fn init(min: f32, max: f32) Interval {
        return .{
            .min = min,
            .max = max,
        };
    }

    pub fn size(self: *const Interval) f32 {
        return self.max - self.min;
    }

    pub fn contains(self: *const Interval, x: f32) bool {
        return self.min <= x and x <= self.max;
    }

    pub fn surrounds(self: *const Interval, x: f32) bool {
        return self.min < x and x < self.max;
    }

    pub fn clamp(self: *const Interval, x: f32) f32 {
        if (x < self.min) return self.min;
        if (x > self.max) return self.max;
        return x;
    }

    pub const empty = Interval.init(math.inf(f32), -std.math.inf(f64));
    pub const univers = Interval.init(-math.inf(f32), std.math.inf(f64));
};

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

fn random_in_unit_sphere() vec3 {
    while (true) {
        const p = random_range(-1, 1);
        if (length_squared(p) < 1) {
            return p;
        }
    }
}

fn random_unit_vector() vec3 {
    return unit_vector(random_in_unit_sphere());
}

fn random_on_hemisphere(normal: vec3) vec3 {
    const on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

fn length_squared(u: vec3) f32 {
    return u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
}

fn random() vec3 {
    return vec3{ rtweekend.random_double(), rtweekend.random_double(), rtweekend.random_double() };
}

fn random_range(min: f32, max: f32) vec3 {
    return vec3{ rtweekend.random_double_range(min, max), rtweekend.random_double_range(min, max), rtweekend.random_double_range(min, max) };
}

fn write_color(out: anytype, c: color) !void {
    const r = c[0];
    const g = c[1];
    const b = c[2];

    const intensity = Interval.init(0.000, 0.999);
    const rbyte = std.math.lossyCast(u8, 256 * intensity.clamp(r));
    const gbyte = std.math.lossyCast(u8, 256 * intensity.clamp(g));
    const bbyte = std.math.lossyCast(u8, 256 * intensity.clamp(b));

    try out.print("{} {} {}\n", .{ rbyte, gbyte, bbyte });
}

const Camera = struct {
    aspect_ratio: f32,
    image_width: u32,
    samples_per_pixel: u32,
    max_depth: u32,

    image_height: u32,
    pixel_samples_scale: f32,
    center: point3,
    pixel00_loc: point3,
    pixel_delta_u: vec3,
    pixel_delta_v: vec3,

    pub fn render(self: *Camera, world: *const Hittable) !void {
        self.initialize();

        const stdout = std.io.getStdOut().writer();
        try stdout.print("P3\n", .{});
        try stdout.print("{} {}\n", .{ self.image_width, self.image_height });
        try stdout.print("255\n", .{});

        for (0..self.image_height) |j| {
            std.log.info("Scanlines remaining: {}", .{(self.image_height - j)});
            for (0..self.image_width) |i| {
                var pixel_color = color{ 0, 0, 0 };
                for (0..self.samples_per_pixel) |_| {
                    const r = get_ray(self, i, j);
                    pixel_color += ray_color(self, r, self.max_depth, world);
                }

                try write_color(&stdout, vec3_splat(self.pixel_samples_scale) * pixel_color);
            }
        }

        std.log.info("image width: {}", .{self.image_width});
        std.log.info("image height: {}", .{self.image_height});
        std.log.info("camera: ({},{},{})", .{ self.center[0], self.center[1], self.center[2] });

        std.log.info("Done\n", .{});
    }

    fn initialize(self: *Camera) void {
        self.image_height = @intFromFloat(@max(1, @as(f32, @floatFromInt(self.image_width)) / self.aspect_ratio));

        self.pixel_samples_scale = 1.0 / @as(f32, @floatFromInt(self.samples_per_pixel));
        // camera
        self.center = point3{ 0, 0, 0 };
        const focal_length: f32 = 1.0;
        const viewport_height: f32 = 2.0;
        const viewport_width: f32 = viewport_height *
            @as(f32, @floatFromInt(self.image_width)) /
            @as(f32, @floatFromInt(self.image_height));

        const viewport_u = vec3{ viewport_width, 0, 0 };
        const viewport_v = vec3{ 0, -viewport_height, 0 };

        // the space between viewport pixels in the x and y direction
        self.pixel_delta_u = viewport_u / vec3_splat(@floatFromInt(self.image_width));
        self.pixel_delta_v = viewport_v / vec3_splat(@floatFromInt(self.image_height));

        // start from the center and get the top left point
        const viewport_upper_left = self.center -
            vec3{ 0, 0, focal_length } -
            viewport_u / vec3_splat(2) -
            viewport_v / vec3_splat(2);
        // inset by half of pixel_delta vectically and horizontally
        self.pixel00_loc = viewport_upper_left +
            vec3_splat(0.5) * (self.pixel_delta_u + self.pixel_delta_v);
    }

    fn get_ray(self: *Camera, i: usize, j: usize) Ray {
        const offset = sample_square();
        const pixel_sample = self.pixel00_loc + ((vec3_splat(@as(f32, @floatFromInt(i)) + offset[0])) * self.pixel_delta_u) + ((vec3_splat(@as(f32, @floatFromInt(j)) + offset[1])) * self.pixel_delta_v);

        const ray_origin = self.center;
        const ray_direction = pixel_sample - ray_origin;

        return Ray.init(ray_origin, ray_direction);
    }

    fn sample_square() vec3 {
        return vec3{ rtweekend.random_double() - 0.5, rtweekend.random_double() - 0.5, 0 };
    }

    fn ray_color(self: *Camera, ray: Ray, depth: u32, world: *const Hittable) color {
        if (depth <= 0) {
            return color{ 0, 0, 0 };
        }
        var rec: HitRecord = undefined;
        if (world.hit(ray, Interval.init(0.001, rtweekend.infinity), &rec)) {
            const direction = random_on_hemisphere(rec.normal);
            return vec3_splat(0.5) * ray_color(self, Ray.init(rec.p, direction), depth - 1, world);
        }

        const unit_direction = unit_vector(ray.direction());
        // map -1 to 1 to 0 to 1
        const a = 0.5 * (unit_direction[1] + 1.0);

        return vec3_splat(1.0 - a) * color{ 1.0, 1.0, 1.0 } +
            vec3_splat(a) * color{ 0.5, 0.7, 1.0 };
    }
};

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
};

const HitRecord = struct {
    p: point3,
    normal: vec3,
    t: f32,
    front_face: bool,

    pub fn set_face_normal(self: *HitRecord, r: Ray, outward_normal: vec3) void {
        self.front_face = dot(r.direction(), outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else -outward_normal;
    }
};

const Hittable = struct {
    ptr: *anyopaque,
    vtab: *const Vtab,
    const Vtab = struct {
        hit: *const fn (self: *anyopaque, r: Ray, ray_t: Interval, rec: *HitRecord) bool,
    };

    pub fn hit(self: *const Hittable, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        return self.vtab.hit(self.ptr, r, ray_t, rec);
    }

    pub fn init(obj: anytype) Hittable {
        const Ptr = @TypeOf(obj);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer);
        assert(PtrInfo.Pointer.size == .One);
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct);
        const impl = struct {
            fn hit(ptr: *anyopaque, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
                const self: Ptr = @ptrCast(@alignCast(ptr));
                return self.hit(r, ray_t, rec);
            }
        };
        return .{
            .ptr = @constCast(obj),
            .vtab = &.{
                .hit = impl.hit,
            },
        };
    }
};

const HittableList = struct {
    objects: ArrayList(Hittable),

    pub fn init(allocator: Allocator) HittableList {
        return .{ .objects = ArrayList(Hittable).init(allocator) };
    }

    pub fn add(self: *HittableList, object: Hittable) !void {
        try self.objects.append(object);
    }

    pub fn hit(self: *HittableList, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        var temp_rec: HitRecord = undefined;
        var hit_anything = false;
        var closest_so_far = ray_t.max;

        for (self.objects.items) |object| {
            if (object.hit(r, Interval.init(ray_t.min, closest_so_far), &temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec.* = temp_rec;
            }
        }

        return hit_anything;
    }
};

const Sphere = struct {
    center: point3,
    radius: f32,
    pub fn init(center: point3, radius: f32) Sphere {
        return Sphere{
            .center = center,
            .radius = @max(0, radius),
        };
    }

    pub fn hit(self: *const Sphere, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        const oc = self.center - r.origin();
        const a = dot(r.direction(), r.direction());
        const h = dot(r.direction(), oc);
        const c = length_squared(oc) - self.radius * self.radius;
        const discriminant = h * h - a * c;

        if (discriminant < 0) {
            return false;
        }

        const sqrtd = math.sqrt(discriminant);

        var root = (h - sqrtd) / a;

        // check both roots
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        rec.*.t = root;
        rec.*.p = r.at(rec.t);
        const outward_normal = (rec.p - self.center) / vec3_splat(self.radius);
        rec.*.set_face_normal(r, outward_normal);

        return true;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var world_list = HittableList.init(allocator);
    var world = Hittable.init(&world_list);
    const sphere1 = Hittable.init(&Sphere.init(point3{ 0, 0, -1 }, 0.5));
    const sphere2 = Hittable.init(&Sphere.init(point3{ 0, -100.5, -1 }, 100));
    try world_list.add(sphere1);
    try world_list.add(sphere2);

    var cam: Camera = undefined;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;

    try cam.render(&world);
}
