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

    pub fn ray_color(self: *const Ray, world: *Hittable) color {
        var rec: HitRecord = undefined;
        if (world.hit(self.*, Interval.init(0, rtweekend.infinity), &rec)) {
            return vec3_splat(0.5) * (rec.normal + color{
                1,
                1,
                1,
            });
        }

        const unit_direction = unit_vector(self.direction());
        // map -1 to 1 to 0 to 1
        const a = 0.5 * (unit_direction[1] + 1.0);

        return vec3_splat(1.0 - a) * color{ 1.0, 1.0, 1.0 } + vec3_splat(a) * color{ 0.5, 0.7, 1.0 };
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
    const aspect_ratio: f32 = 16.0 / 9.0;
    const image_width: u32 = 400;
    const image_height = @max(1, @as(u32, @intFromFloat(@floor(image_width /
        aspect_ratio))));

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var world_list = HittableList.init(allocator);
    var world = Hittable.init(&world_list);
    const sphere1 = Hittable.init(&Sphere.init(point3{ 0, 0, -1 }, 0.5));
    const sphere2 = Hittable.init(&Sphere.init(point3{ 0, -100.5, -1 }, 100));
    try world_list.add(sphere1);
    try world_list.add(sphere2);

    // camera
    const focal_length: f32 = 1.0;
    const viewport_height: f32 = 2.0;
    const viewport_width = viewport_height * image_width / image_height;
    const camera_center = point3{ 0, 0, 0 };

    const viewport_u = vec3{ viewport_width, 0, 0 };
    const viewport_v = vec3{ 0, -viewport_height, 0 };

    // the space between viewport pixels in the x and y direction
    const pixel_delta_u = viewport_u / vec3_splat(image_width);
    const pixel_delta_v = viewport_v / vec3_splat(image_height);

    // start from the center and get the top left point
    const viewport_upper_left = camera_center - vec3{ 0, 0, focal_length } - viewport_u / vec3_splat(2) - viewport_v / vec3_splat(2);
    // inset by half of pixel_delta vectically and horizontally
    const pixel00_loc = viewport_upper_left + vec3_splat(0.5) * (pixel_delta_u + pixel_delta_v);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("P3\n", .{});
    try stdout.print("{} {}\n", .{ image_width, image_height });
    try stdout.print("255\n", .{});

    for (0..image_height) |j| {
        std.log.info("Scanlines remaining: {}", .{(image_height - j)});
        for (0..image_width) |i| {
            // each actual image pixel corresponds to a viewport pixel
            const pixel_center = pixel00_loc +
                (vec3_splat(@floatFromInt(i)) * pixel_delta_u) + (vec3_splat(@floatFromInt(j)) * pixel_delta_v);

            const ray_direction = pixel_center - camera_center;
            const r = Ray.init(camera_center, ray_direction);

            const pixel_color = r.ray_color(&world);

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
