const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const math = std.math;
const assert = std.debug.assert;
const utils = @import("utils.zig");

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

// computes the dot product of two vectors u and v
fn dot(u: vec3, v: vec3) f32 {
    return @reduce(.Add, u * v);
}

// computes the cross product of two vectors u and v
fn cross(u: vec3, v: vec3) vec3 {
    return vec3{ u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0] };
}

// returns a 3D vector out of a scalar
fn vec3_splat(scalar: f32) vec3 {
    return @as(vec3, @splat(scalar));
}

fn length(u: vec3) f32 {
    return math.sqrt(length_squared(u));
}

// normalizes vector u
fn unit_vector(u: vec3) vec3 {
    return u / vec3_splat(length(u));
}

// generates a random vector on a unit disk by rejecting samples from a unit square
fn random_in_unit_disk() vec3 {
    while (true) {
        const p = vec3{ utils.random_double_range(-1, 1), utils.random_double_range(-1, 1), 0 };
        if (length_squared(p) < 1) {
            return p;
        }
    }
}

// generates a random vector on a unit sphere by rejecting samples from a unit cube
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

// generates a random vector on the same hemisphere as normal
fn random_on_hemisphere(normal: vec3) vec3 {
    const on_unit_sphere = random_unit_vector();
    // check if similarly oriented with the normal, otherwise reverse the direction
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

// reflected vector is v - 2b where b is the projection of v onto n
// returns v mirrored across the plane perpendicular to n
fn reflect(v: vec3, n: vec3) vec3 {
    return v - vec3_splat(2 * dot(v, n)) * n;
}

// this function implements Snell's law for refraction:
// - calculates the cosine of the angle between the incident ray and the normal
// - computes the perpendicular component of the refracted ray
// - computes the parallel component of the refracted ray
// - combines the components to get the final refracted vector
//
// note: this implementation assumes total internal reflection is handled elsewhere
fn refract(uv: vec3, n: vec3, etai_over_etat: f32) vec3 {
    // calculates the cosine of the angle between the incident ray and the normal
    // clamped to 1.0 to handle numerical imprecision
    const cos_theta = @min(dot(-uv, n), 1.0);

    // r_out_perp = etai_over_etat * (uv + cos_theta * n)
    const r_out_perp = vec3_splat(etai_over_etat) * (uv + vec3_splat(cos_theta) * n);

    // r_out_parallel = -sqrt(1 - |r_out_perp|^2) * n (pythagorean theorem)
    // sign inverted to point away from the normal
    const r_out_parallel = vec3_splat(-math.sqrt(@abs(1.0 - length_squared(r_out_perp)))) * n;

    // combine components to get the refracted vector
    return r_out_perp + r_out_parallel;
}

fn length_squared(u: vec3) f32 {
    return u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
}

fn near_zero(u: vec3) bool {
    const s = 1e-8;
    return @abs(u[0]) < s and @abs(u[1]) < s and @abs(u[2]) < s;
}

fn random() vec3 {
    return vec3{ utils.random_double(), utils.random_double(), utils.random_double() };
}

fn random_range(min: f32, max: f32) vec3 {
    return vec3{ utils.random_double_range(min, max), utils.random_double_range(min, max), utils.random_double_range(min, max) };
}

fn linear_to_gamma(linear_component: f32) f32 {
    if (linear_component > 0) {
        return math.sqrt(linear_component);
    }

    return 0;
}

fn write_color(out: anytype, c: color) !void {
    var r = c[0];
    var g = c[1];
    var b = c[2];

    // apply gamma correction
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    const intensity = Interval.init(0.000, 0.999);
    const rbyte = std.math.lossyCast(u8, 256 * intensity.clamp(r));
    const gbyte = std.math.lossyCast(u8, 256 * intensity.clamp(g));
    const bbyte = std.math.lossyCast(u8, 256 * intensity.clamp(b));

    try out.print("{} {} {}\n", .{ rbyte, gbyte, bbyte });
}

const Camera = struct {
    // ########### configurable properties ##########
    aspect_ratio: f32, // aspect ratio of the image and viewport
    image_width: u32,

    samples_per_pixel: u32, // amount of rays to shoot per pixel
    max_depth: u32, // maximum number of ray "bounces"
    vfov: f32 = 90, // vertical fov in degrees, viewport_height is computed from this

    // aperture. larger angle will result in more background blur
    // setting to 0 will produce perfectly sharp image
    defocus_angle: f32,

    // distance at which objects are in focus, objects closer or farther
    // will appear progressively more blurred
    focus_dist: f32,

    // ########## computed properties (do not change) ##########
    image_height: u32,
    pixel_samples_scale: f32,
    center: point3, // camera center

    // viewport information. (0, 0) starts at the top left
    pixel00_loc: point3, // (0, 0) pixel on the plane of the viewport
    pixel_delta_u: vec3, // pixel spacing in the u direction (horizontal)
    pixel_delta_v: vec3, // pixel spacing in the v direction (vertical)

    lookfrom: point3, // position of the camera
    lookat: point3, // position of the viewport's center

    // - arbitrary "up" vector from which to compute camera basis vectors
    // - only constraint is for it not to be parallel to w
    // - the cross product of vup and w is used to find u
    vup: vec3,

    // camera basis vectors
    u: vec3,
    v: vec3,
    w: vec3,

    // - vectors defining the u and v axes of the camera's circular defocus disk
    // - disk is orthogonal to the basis direction w
    // - computed using focus_dist and defocus_angle, where the disk is the
    // base of cone with apex focust_dist and angle defocus_angle
    // - rays are randomly sampled from this disk and shot through the pixel on
    // the viewport creating the DoF effect
    // - smaller defocus_angle = smaller disk = less blur
    // - larger focus_dist = larger disk = more blur
    defocus_disk_v: vec3,
    defocus_disk_u: vec3,

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
                    // accumulate color contribution per ray sample
                    pixel_color += ray_color(self, r, self.max_depth, world);
                }

                // write average of accumulated color information
                try write_color(&stdout, vec3_splat(self.pixel_samples_scale) * pixel_color);
            }
        }

        std.log.info("image width: {}", .{self.image_width});
        std.log.info("image height: {}", .{self.image_height});
        std.log.info("camera: ({},{},{})", .{ self.center[0], self.center[1], self.center[2] });

        std.log.info("Done\n", .{});
    }

    // initializes the camera. always called before render
    fn initialize(self: *Camera) void {
        self.image_height = @intFromFloat(@max(1, @as(f32, @floatFromInt(self.image_width)) / self.aspect_ratio));

        self.pixel_samples_scale = 1.0 / @as(f32, @floatFromInt(self.samples_per_pixel));
        // camera
        self.center = self.lookfrom;

        // viewport
        const theta = utils.degrees_to_radians(self.vfov);
        // - the ratio t = 1/2 viewport_height / focus_dist
        const t = math.tan(theta / 2);
        // - solving for viewport_height: viewport_height = 2 * t * focus_dist
        const viewport_height: f32 = 2 * t * self.focus_dist;
        // - the viewport dimensions is computed to have the same aspect ratio
        // as the resulting image
        const viewport_width: f32 = viewport_height *
            @as(f32, @floatFromInt(self.image_width)) /
            @as(f32, @floatFromInt(self.image_height));

        // right-hand basis vectors
        // - w points away from the viewing direction (towards the camera)
        self.w = unit_vector(self.lookfrom - self.lookat);
        // the unit vector orthogonal to both vup and w representing "right"
        self.u = unit_vector(cross(self.vup, self.w));
        // the unit vector orthogonal to both w and u representing "up"
        self.v = cross(self.w, self.u);

        // vector in the u (right) direction that spans the viewport_width
        const viewport_u = vec3_splat(viewport_width) * self.u;
        // vector in the -v (down) direction that spans the viewport_height
        const viewport_v = vec3_splat(viewport_height) * -self.v;

        // the space between viewport pixels in the u and -v direction
        self.pixel_delta_u = viewport_u / vec3_splat(@floatFromInt(self.image_width));
        self.pixel_delta_v = viewport_v / vec3_splat(@floatFromInt(self.image_height));

        // 1. start from the center and go focus_dist distance in the -w direction.
        //    this is the center of the viewport.
        // 2. go half the width of the viewport in the -viewport_u direction (left)
        // 3. go half the height of the viewport in the -viewport_v direction (up)
        const viewport_upper_left = self.center -
            vec3_splat(self.focus_dist) * self.w -
            viewport_u / vec3_splat(2) -
            viewport_v / vec3_splat(2);

        // inset by half of pixel_delta vectically and horizontally
        self.pixel00_loc = viewport_upper_left +
            vec3_splat(0.5) * (self.pixel_delta_u + self.pixel_delta_v);

        // defocus_disk plane is computed in the same spirit as the viewport except
        // we only compute a radius
        const defocus_radius = self.focus_dist * math.tan(utils.degrees_to_radians(self.defocus_angle / 2));
        self.defocus_disk_u = self.u * vec3_splat(defocus_radius);
        self.defocus_disk_v = self.v * vec3_splat(defocus_radius);
    }

    // sample a ray for the i, j pixel
    fn get_ray(self: *Camera, i: usize, j: usize) Ray {
        // offset where the ray actually shoots through by a random amount within
        // the pixels center in the range [-0.5, 0.5] in both the u, v directions
        // to achieve anti-aliasing
        const offset = sample_square();
        const pixel_sample = self.pixel00_loc + ((vec3_splat(@as(f32, @floatFromInt(i)) + offset[0])) * self.pixel_delta_u) + ((vec3_splat(@as(f32, @floatFromInt(j)) + offset[1])) * self.pixel_delta_v);

        // if defocus_angle is defined then shoot the ray from a random point
        // on the defocus disk to create DoF effect,
        // otherwise shoot the ray from the camera center
        const ray_origin = if (self.defocus_angle <= 0) self.center else self.defocus_disk_sample();
        const ray_direction = pixel_sample - ray_origin;

        return Ray.init(ray_origin, ray_direction);
    }

    // generate a random vector in the u, v plane in the range [-0.5, 0.5]
    fn sample_square() vec3 {
        return vec3{ utils.random_double() - 0.5, utils.random_double() - 0.5, 0 };
    }

    // generate a point in the u,v unit disk plane centered at the camera's center
    fn defocus_disk_sample(self: *const Camera) point3 {
        const p = random_in_unit_disk();
        return self.center + (vec3_splat(p[0]) * self.defocus_disk_u) + (vec3_splat(p[1]) * self.defocus_disk_v);
    }

    // - calculates the color of a ray in the scene
    // - uses recursive ray tracing with a maximum depth to limit bounces
    // - the ray color is a product of attenuations < 1 until ray color
    //   reaches max depth or fails to scatter. this multiplicative factor
    //   has the effect of reducing the intensity of the ray per bounce
    fn ray_color(self: *Camera, ray: Ray, depth: u32, world: *const Hittable) color {
        // if ray is exhausted (bounced max times), return black
        if (depth <= 0) {
            return color{ 0, 0, 0 };
        }

        var rec: HitRecord = undefined;
        // if the ray hits an object in our scene,
        //   - attempt to scatter the ray and compute attenuation
        //   - if successful, recursively trace the scattered ray
        // 0.001 offset to mitigate "shadow acne" due to floating point imprecision
        if (world.hit(ray, Interval.init(0.001, utils.infinity), &rec)) {
            var scattered: Ray = undefined;
            var attenuation: color = undefined;
            if (rec.mat.scatter(ray, &rec, &attenuation, &scattered)) {
                return attenuation * self.ray_color(scattered, depth - 1, world);
            }
            return color{ 0, 0, 0 };
        }

        // if the ray misses all objects,
        // return a background color based on the ray's vertical direction
        const unit_direction = unit_vector(ray.direction());

        // map -1 to 1 to 0 to 1
        const a = 0.5 * (unit_direction[1] + 1.0);

        // background is a simple gradient from white to light blue
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

    // returns a point along the ray at distance t from the ray origin
    pub fn at(self: *const Ray, t: f32) point3 {
        return self.orig + vec3_splat(t) * self.dir;
    }
};

// store information after a ray hits an object
const HitRecord = struct {
    p: point3, // point p in which the ray hit an object
    normal: vec3, // normal of the object at point p
    mat: Material, // material of the object hit
    t: f32, // parameter t representing distance along the ray on point of intersection
    front_face: bool, //  whether the ray is opposite the normal.

    // determines whether the ray hit the object from outside or inside
    pub fn set_face_normal(self: *HitRecord, r: Ray, outward_normal: vec3) void {
        // if dot product between r and the outward normal is negative
        // then they are oriented away from each other which means the ray
        // hit from the outside, otherwise they are in the same orientation
        // and the ray hit from the inside (e.g. when a ray passes through
        // glass from the inside)
        self.front_face = dot(r.direction(), outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else -outward_normal;
    }
};

// mom can we have interface
// mom: we have interface at home
// *the interface at home*
// material information for an object to determin how a ray scatters when hit
const Material = struct {
    ptr: *anyopaque,
    vtab: *const Vtab,
    const Vtab = struct {
        scatter: *const fn (self: *const anyopaque, r_in: Ray, rec: *HitRecord, attenuation: *color, scattered: *Ray) bool,
    };

    pub fn scatter(self: *const Material, r_in: Ray, rec: *HitRecord, attenuation: *color, scattered: *Ray) bool {
        return self.vtab.scatter(self.ptr, r_in, rec, attenuation, scattered);
    }

    pub fn init(obj: anytype) Material {
        const Ptr = @TypeOf(obj);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer);
        assert(PtrInfo.Pointer.size == .One);
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct);
        const impl = struct {
            fn scatter(ptr: *const anyopaque, r_in: Ray, rec: *HitRecord, attenuation: *color, scattered: *Ray) bool {
                const self: Ptr = @ptrCast(@alignCast(@constCast(ptr)));
                return self.scatter(r_in, rec, attenuation, scattered);
            }
        };
        return .{
            .ptr = @constCast(obj),
            .vtab = &.{
                .scatter = impl.scatter,
            },
        };
    }
};

// matte surface with diffused reflectance
const Lambertian = struct {
    albedo: color, // reflectance

    pub fn init(albedo: color) Lambertian {
        return .{ .albedo = albedo };
    }

    pub fn scatter(self: *const Lambertian, r_in: Ray, rec: *HitRecord, attenuation: *color, scattered: *Ray) bool {
        _ = r_in;
        // offset the normal by a random unit direction to achieve
        // lamebertian reflection
        var scatter_direction = rec.normal + random_unit_vector();

        // handle the case where the normal and the generated direction is
        // opposite to each other
        if (near_zero(scatter_direction)) {
            scatter_direction = rec.normal;
        }
        scattered.* = Ray.init(rec.p, scatter_direction);
        attenuation.* = self.albedo;
        return true;
    }
};

// reflective surface with fuzzy reflection
const Metal = struct {
    albedo: color,
    fuzz: f32, // fuzzy reflection factor (metal is not a perfect mirror)

    pub fn init(albedo: color, fuzz: f32) Metal {
        return .{ .albedo = albedo, .fuzz = if (fuzz < 1) fuzz else 1 };
    }

    pub fn scatter(self: *const Metal, r_in: Ray, rec: *HitRecord, attenuation: *color, scattered: *Ray) bool {
        var reflected = reflect(r_in.direction(), rec.normal);
        // offset the reflected vector by a random unit vector scaled by fuzz
        reflected = unit_vector(reflected) + (vec3_splat(self.fuzz) * random_unit_vector());
        scattered.* = Ray.init(rec.p, reflected);
        attenuation.* = self.albedo;
        return true;
    }
};

// clear materials that can either reflect or refract a ray
const Dialectric = struct {
    refraction_index: f32, // amount that describes how much light bends
    pub fn init(refraction_index: f32) Dialectric {
        return .{ .refraction_index = refraction_index };
    }

    pub fn scatter(self: *const Dialectric, r_in: Ray, rec: *HitRecord, attenuation: *color, scattered: *Ray) bool {
        attenuation.* = color{ 1.0, 1.0, 1.0 };
        // the refractive index is the ratio between the refraction_index eta of
        // the surrounding material and the refraction index eta` of the enclosed
        // material.
        // since the refraction index of air is ≈1, the refractive index for a ray
        // going inside the dialectric is 1 / refraction_index, otherwise it is
        // refraction_index / 1 = refraction_index
        const ri = if (rec.front_face) (1.0 / self.refraction_index) else self.refraction_index;
        const unit_direction = unit_vector(r_in.direction());
        // cosine of the angle between the unit direction of the incident vector
        // and the normal, sign is flipped to make sure unit direction is oriented
        // the same way as the normal
        const cos_theta = @min(dot(-unit_direction, rec.normal), 1.0);
        const sin_theta = math.sqrt(1.0 - cos_theta * cos_theta);

        // account for total internal reflection, when refraction according to
        // Snell's law is impossible i.e. when sin(θ') = ri * sin(θ) > 1.0
        // since sin(x) is never greater than 1, just reflect
        const cannot_refract = ri * sin_theta > 1.0;
        var direction: vec3 = undefined;
        if (cannot_refract or reflectance(cos_theta, ri) > utils.random_double()) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, ri);
        }
        scattered.* = Ray.init(rec.p, direction);
        return true;
    }

    // Schlick's approximation to vary reflectivity by angle
    fn reflectance(cosine: f32, refraction_index: f32) f32 {
        var r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * math.pow(f32, (1 - cosine), 5);
    }
};

// mom can we have interface
// mom: we have interface at home
// *the interface at home*
// any object that can report when a ray has hit it
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

// a wrapper list around a bunch of Hittables that reports the closest object
// a ray has hit
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

        // find the closest object the ray has hit
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

    mat: Material,
    pub fn init(center: point3, radius: f32, mat: Material) Sphere {
        return Sphere{
            .center = center,
            .radius = @max(0, radius),
            .mat = mat,
        };
    }

    pub fn hit(self: *const Sphere, r: Ray, ray_t: Interval, rec: *HitRecord) bool {
        // you can derive for yourself using the ff facts:
        // - start with the sphere equation (Cx-x)^2 + (Cy-y)^2 + (Cz-z)^2 = r^2
        //   or in vector form (C - P)·(C - P) = r^2, where P is (x, y, z)
        // - the P we are interested in is the ray P(t) = Q + td where Q is the
        //   origin and d is the direction vector. our goal is to find which t
        //   parameterizes the ray that satisfies the sphere equation.
        // - Plug P(t) to get (C - (Q + td))·(C - (Q + td)) = r^2.
        // - do some algebra till you arrive at an equation in standard form
        //   at^2 + bt + c = 0
        // - realize that a = d·d, b = -2d·(C - Q) and c = (C - Q)·(C - Q) - r^2
        // - to simplify the quadratic formula, set b = 2h and solve for h to get
        //   h = b/-2
        // - the discrimant will then be h^2 - a * c, which tells you the existence
        // of roots.
        //   if the discriminant is:
        //   - negative, then there are 0 roots and the ray does not intersect.
        //   - 0, then there is 1 root and the ray intersects the sphere.
        //   - positive,  there are two roots and the ray intersects the sphere in two
        //   places.
        // - simplified quadratic formula is t = (h +- sqrt(h^2 - ac)) / a
        const oc = self.center - r.origin();
        const a = dot(r.direction(), r.direction());
        const h = dot(r.direction(), oc);
        // the dot product of a vector v with itself is its squared length
        const c = length_squared(oc) - self.radius * self.radius;
        const discriminant = h * h - a * c;

        if (discriminant < 0) {
            return false;
        }

        const sqrtd = math.sqrt(discriminant);

        var root = (h - sqrtd) / a;

        // check whether both roots exist and are in our min and max ray range
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        rec.*.t = root;
        rec.*.p = r.at(rec.t);
        // the normal of a sphere at point p is the vector shooting from the center to
        // point p. since the radius is the length of that vector, dividing
        // by the radius will normalize
        const outward_normal = (rec.p - self.center) / vec3_splat(self.radius);
        rec.*.set_face_normal(r, outward_normal);
        rec.*.mat = self.mat;

        return true;
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var world_list = HittableList.init(allocator);
    var world = Hittable.init(&world_list);

    const ground_material = Material.init(&Lambertian.init(color{ 0.5, 0.5, 0.5 }));
    const ground_sphere = Hittable.init(&Sphere.init(point3{ 0, -1000, -1 }, 1000, ground_material));
    try world_list.add(ground_sphere);

    var a: i32 = -11;
    while (a < 11) : (a += 1) {
        var b: i32 = -11;
        while (b < 11) : (b += 1) {
            const choose_mat = utils.random_double();
            // create a slight random offset so spheres arent arranged in a grid
            const center = point3{ @as(f32, @floatFromInt(a)) + 0.9 * utils.random_double(), 0.2, @as(f32, @floatFromInt(b)) + 0.9 * utils.random_double() };

            // dont make small spheres too close to our big metal sphere
            if (length(center - point3{ 4, 0.2, 0 }) > 0.9) {
                const sphere_material: *Material = try allocator.create(Material);
                const sphere: *Sphere = try allocator.create(Sphere);
                if (choose_mat < 0.8) {
                    const albedo: color = random() * random();
                    const lambertian: *Lambertian = try allocator.create(Lambertian);
                    lambertian.* = Lambertian.init(albedo);
                    sphere_material.* = Material.init(lambertian);
                    sphere.* = Sphere.init(center, 0.2, sphere_material.*);
                    try world_list.add(Hittable.init(sphere));
                } else if (choose_mat < 0.95) {
                    const albedo: color = random_range(0.5, 1);
                    const fuzz = utils.random_double_range(0, 0.5);
                    const metal: *Metal = try allocator.create(Metal);
                    metal.* = Metal.init(albedo, fuzz);
                    sphere_material.* = Material.init(metal);
                    sphere.* = Sphere.init(center, 0.2, sphere_material.*);
                    try world_list.add(Hittable.init(sphere));
                } else {
                    const dialectric: *Dialectric = try allocator.create(Dialectric);
                    dialectric.* = Dialectric.init(1.5);
                    sphere_material.* = Material.init(dialectric);
                    sphere.* = Sphere.init(center, 0.2, sphere_material.*);
                    try world_list.add(Hittable.init(sphere));
                }
            }
        }
    }

    const material1 = Material.init(&Dialectric.init(1.5));
    try world_list.add(Hittable.init(&Sphere.init(point3{ 0, 1, 0 }, 1.0, material1)));

    const material2 = Material.init(&Lambertian.init(color{ 0.4, 0.2, 0.1 }));
    try world_list.add(Hittable.init(&Sphere.init(point3{ -4, 1, 0 }, 1.0, material2)));

    const material3 = Material.init(&Metal.init(color{ 0.7, 0.6, 0.5 }, 0.0));
    try world_list.add(Hittable.init(&Sphere.init(point3{ 4, 1, 0 }, 1.0, material3)));

    var cam: Camera = undefined;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 1200;
    cam.samples_per_pixel = 500;
    cam.max_depth = 50;
    cam.vfov = 20;
    cam.lookfrom = point3{ 13, 2, 3 };
    cam.lookat = point3{ 0, 0, 0 };
    cam.vup = point3{ 0, 1, 0 };

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    try cam.render(&world);
}
