// Minimal reader for NumPy .npy arrays and .npz archives.
//
// np.savez writes an uncompressed ZIP (ZIP_STORED) of .npy files, so we can read
// it with no zlib and no third-party deps: walk the ZIP local headers, and parse
// each stored .npy. Supports little-endian float32 (<f4), int32 (<i4), and
// float64 (<f8), C-order, which is all the exporter emits.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::npy {

struct Array {
  std::string dtype;            // "<f4", "<i4", "<f8"
  std::vector<size_t> shape;    // row-major
  std::vector<char> bytes;      // raw little-endian data

  size_t numel() const {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
  }
  const float* f32() const { return reinterpret_cast<const float*>(bytes.data()); }
  const int32_t* i32() const { return reinterpret_cast<const int32_t*>(bytes.data()); }
  const double* f64() const { return reinterpret_cast<const double*>(bytes.data()); }
};

namespace detail {

// Little-endian byte readers. The uint8_t casts stop a high bit in a signed char
// from sign-extending and corrupting the value.
inline uint16_t rd16(const char* p) {
  return static_cast<uint8_t>(p[0]) | (static_cast<uint8_t>(p[1]) << 8);
}
inline uint32_t rd32(const char* p) {
  return static_cast<uint8_t>(p[0]) | (static_cast<uint8_t>(p[1]) << 8) |
         (static_cast<uint8_t>(p[2]) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}
inline uint64_t rd64(const char* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<uint64_t>(static_cast<uint8_t>(p[i])) << (8 * i);
  return v;
}

// Parse a .npy blob (magic + header dict + data) into an Array.
inline Array parse_npy(const char* data, size_t size) {
  if (size < 10 || std::memcmp(data, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("bad .npy magic");
  const uint8_t major = static_cast<uint8_t>(data[6]);
  size_t header_len, header_off;
  if (major == 1) {  // v1.0: 2-byte header length
    header_len = rd16(data + 8);
    header_off = 10;
  } else {  // v2.0+: 4-byte header length
    header_len = rd32(data + 8);
    header_off = 12;
  }
  std::string header(data + header_off, header_len);
  size_t data_off = header_off + header_len;

  Array a;
  auto dpos = header.find("'descr'");
  auto q1 = header.find('\'', header.find(':', dpos) + 1);
  auto q2 = header.find('\'', q1 + 1);
  a.dtype = header.substr(q1 + 1, q2 - q1 - 1);

  if (header.find("'fortran_order': True") != std::string::npos)
    throw std::runtime_error("fortran_order arrays are not supported");

  auto sp = header.find("'shape'");
  auto lp = header.find('(', sp);
  auto rp = header.find(')', lp);
  std::string dims = header.substr(lp + 1, rp - lp - 1);
  size_t pos = 0;
  while (pos < dims.size()) {  // pull the integers out of "(256, 64)"
    while (pos < dims.size() && (dims[pos] == ' ' || dims[pos] == ',')) ++pos;
    if (pos >= dims.size()) break;
    size_t end = pos;
    while (end < dims.size() && dims[end] >= '0' && dims[end] <= '9') ++end;
    if (end > pos) a.shape.push_back(std::stoul(dims.substr(pos, end - pos)));
    pos = end;
  }

  size_t itemsize = static_cast<size_t>(a.dtype.back() - '0');  // '<f4' -> 4
  size_t nbytes = a.numel() * itemsize;
  a.bytes.assign(data + data_off, data + data_off + nbytes);
  return a;
}

}  // namespace detail

// Read a .npz (uncompressed ZIP of .npy files) into {name -> Array}.
inline std::map<std::string, Array> load_npz(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});

  std::map<std::string, Array> out;
  size_t off = 0;
  // Each stored entry starts with the local file header signature PK\x03\x04.
  // Loop stops at the central directory (a different signature).
  while (off + 4 <= buf.size() && detail::rd32(buf.data() + off) == 0x04034b50) {
    const char* h = buf.data() + off;
    uint16_t method = detail::rd16(h + 8);
    uint16_t name_len = detail::rd16(h + 26);
    uint16_t extra_len = detail::rd16(h + 28);
    std::string name(h + 30, name_len);
    if (method != 0) throw std::runtime_error("compressed .npz entry not supported: " + name);

    // Data size. 0xFFFFFFFF is the ZIP64 sentinel; the real 64-bit size lives in
    // the extra field under id 0x0001 (uncompressed size, then compressed size).
    uint64_t data_size = detail::rd32(h + 18);
    if (data_size == 0xFFFFFFFFu) {
      const char* ex = h + 30 + name_len;
      for (size_t ei = 0; ei + 4 <= extra_len;) {
        uint16_t id = detail::rd16(ex + ei);
        uint16_t sz = detail::rd16(ex + ei + 2);
        if (id == 0x0001) {
          data_size = detail::rd64(ex + ei + 4 + (sz >= 16 ? 8 : 0));
          break;
        }
        ei += 4 + sz;
      }
    }

    size_t data_off = off + 30 + name_len + extra_len;
    if (name.size() > 4 && name.substr(name.size() - 4) == ".npy")
      name = name.substr(0, name.size() - 4);
    out.emplace(name, detail::parse_npy(buf.data() + data_off, data_size));
    off = data_off + data_size;
  }
  if (out.empty()) throw std::runtime_error("no entries parsed from " + path);
  return out;
}

}  // namespace engine::npy
