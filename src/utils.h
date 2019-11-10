#ifndef UTILS_H
#define UTILS_H

#include <math.h>


class Vec2
{
public:
    float x, y;
    Vec2() = default;
    Vec2(float vx, float vy)
    {
        x = vx;
        y = vy;
    }
    static inline float dot(const Vec2 & v0, const Vec2 & v1)
    {
        return v0.x * v1.x + v0.y * v1.y;
    }
    inline float & operator [] (int i)
    {
        return ((float*)this)[i];
    }
    inline Vec2 operator * (float s) const
    {
        Vec2 rs;
        rs.x = x * s;
        rs.y = y * s;
        return rs;
    }
    inline Vec2 operator * (const Vec2 &vin) const
    {
        Vec2 rs;
        rs.x = x * vin.x;
        rs.y = y * vin.y;
        return rs;
    }
    inline Vec2 operator + (const Vec2 &vin) const
    {
        Vec2 rs;
        rs.x = x + vin.x;
        rs.y = y + vin.y;
        return rs;
    }
    inline Vec2 operator - (const Vec2 &vin) const
    {
        Vec2 rs;
        rs.x = x - vin.x;
        rs.y = y - vin.y;
        return rs;
    }
    inline Vec2 operator -() const
    {
        Vec2 rs;
        rs.x = -x;
        rs.y = -y;
        return rs;
    }
    inline Vec2 & operator += (const Vec2 & vin)
    {
        x += vin.x;
        y += vin.y;
        return *this;
    }
    inline Vec2 & operator -= (const Vec2 & vin)
    {
        x -= vin.x;
        y -= vin.y;
        return *this;
    }
    Vec2 & operator = (float v)
    {
        x = y = v;
        return *this;
    }
    inline Vec2 & operator *= (float s)
    {
        x *= s;
        y *= s;
        return *this;
    }
    inline Vec2 & operator *= (const Vec2 & vin)
    {
        x *= vin.x;
        y *= vin.y;
        return *this;
    }
    inline Vec2 normalize()
    {
        float len = sqrt(x*x + y*y);
        float invLen = 1.0f / len;
        Vec2 rs;
        rs.x = x * invLen;
        rs.y = y * invLen;
        return rs;
    }
    inline float length()
    {
        return sqrt(x*x + y*y);
    }
};

#endif
