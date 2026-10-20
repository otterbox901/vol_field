//
// src/field/cli.cpp - command-line flags shared by the driver binaries.
//

#include "vf/cli.hpp"

#include <arpa/inet.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace vf {
namespace {

/// Every flag taking an unsigned value. `usage` is both the name we match on
/// and the help text's left column, so the two can never drift apart.
struct U32Flag {
    const char* usage;  ///< "--queues N": the flag, then its value placeholder
    const char* help;
    uint32_t RunConfig::* field;
};

constexpr U32Flag kU32Flags[] = {
    {"--queues N", "receive blocks, one per queue (default 1)", &RunConfig::queues},
    {"--duration MS", "how long to run, 0 = until Ctrl-C (default 10000)",
     &RunConfig::duration_ms},
    {"--snapshot-ms MS", "surface pass + snapshot period (default 100)",
     &RunConfig::snapshot_ms},
};

/// True when `a` is this flag, i.e. it equals `usage` up to the placeholder.
bool flag_matches(const char* a, const char* usage) {
    const size_t n = std::strcspn(usage, " ");
    return std::strncmp(a, usage, n) == 0 && a[n] == '\0';
}

}  // namespace

void print_common_usage() {
    const auto line = [](const char* flag, const char* help) {
        std::printf("  %-*s%s\n", kUsageWidth, flag, help);
    };
    line("--underlying SYM[,..]", "underlying(s) to build the surface for (required, up to 8)");
    line("--date YYYY-MM-DD", "trade date, for time to expiry (default: today)");
    line("--rate R", "continuously compounded rate, e.g. 0.04 (default 0)");
    for (const U32Flag& f : kU32Flags) line(f.usage, f.help);
    line("--shm NAME", "snapshot in /dev/shm/NAME, or a path on any tmpfs (default vol_field)");
    line("--verbose", "print device and allocation details");
    std::printf("  --help\n");
}

bool take_u64(int argc, char** argv, int& i, uint64_t& out) {
    if (i + 1 >= argc) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(argv[++i], &end, 10);
    if (end == argv[i] || *end != '\0' || errno != 0 || argv[i][0] == '-') return false;
    out = v;
    return true;
}

bool take_double(int argc, char** argv, int& i, double& out) {
    if (i + 1 >= argc) return false;
    char* end = nullptr;
    const double v = std::strtod(argv[++i], &end);
    if (end == argv[i] || *end != '\0') return false;
    out = v;
    return true;
}

bool take_str(int argc, char** argv, int& i, const char*& out) {
    if (i + 1 >= argc) return false;
    out = argv[++i];
    return true;
}

bool parse_date(const char* s, int32_t& days) {
    int y = 0;
    unsigned m = 0, d = 0;
    char tail = 0;
    if (std::sscanf(s, "%4d-%2u-%2u%c", &y, &m, &d, &tail) != 3) return false;
    if (y < 2000 || y > 2099 || m < 1 || m > 12 || d < 1 || d > 31) return false;
    days = days_from_civil(y, m, d);
    return true;
}

bool parse_endpoint(const char* s, uint32_t& ip, uint16_t& port) {
    const char* colon = std::strrchr(s, ':');
    const char* port_str = colon ? colon + 1 : s;
    ip = 0;
    if (colon && colon != s) {
        const std::string host(s, colon);
        in_addr a{};
        if (inet_pton(AF_INET, host.c_str(), &a) != 1) return false;
        ip = ntohl(a.s_addr);
    }
    char* end = nullptr;
    const unsigned long p = std::strtoul(port_str, &end, 10);
    if (end == port_str || *end != '\0' || p == 0 || p > 65535) return false;
    port = static_cast<uint16_t>(p);
    return true;
}

bool parse_endpoints(const char* s, std::vector<Endpoint>& out) {
    const std::vector<std::string> parts = split_list(s);
    for (const std::string& part : parts) {
        Endpoint e{};
        if (!parse_endpoint(part.c_str(), e.ip, e.port)) return false;
        out.push_back(e);
    }
    return !parts.empty();
}

std::string endpoint_string(const Endpoint& e) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%u.%u.%u.%u:%u", e.ip >> 24, (e.ip >> 16) & 0xFF,
                  (e.ip >> 8) & 0xFF, e.ip & 0xFF, e.port);
    return e.ip ? buf : ":" + std::to_string(e.port);
}

std::vector<std::string> split_list(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (; ; ++s) {
        if (*s == ',' || *s == '\0') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*s == '\0') break;
        } else {
            cur.push_back(*s);
        }
    }
    return out;
}

FlagResult parse_common_flag(int argc, char** argv, int& i, RunConfig& cfg) {
    const char* a = argv[i];
    const char* s = nullptr;
    bool ok = true;

    if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
        return FlagResult::kHelp;
    } else if (!std::strcmp(a, "--verbose")) {
        cfg.verbose = true;
    } else if (!std::strcmp(a, "--underlying")) {
        ok = take_str(argc, argv, i, s);
        if (ok) {
            for (const std::string& u : split_list(s)) cfg.underlyings.push_back(u);
        }
    } else if (!std::strcmp(a, "--date")) {
        ok = take_str(argc, argv, i, s) && parse_date(s, cfg.trade_days);
        if (ok) cfg.date = s;
    } else if (!std::strcmp(a, "--rate")) {
        ok = take_double(argc, argv, i, cfg.rate) && cfg.rate > -1.0 && cfg.rate < 1.0;
    } else if (!std::strcmp(a, "--shm")) {
        ok = take_str(argc, argv, i, s) && *s;
        if (ok) cfg.shm = s;
    } else {
        const U32Flag* flag = nullptr;
        for (const U32Flag& f : kU32Flags) {
            if (flag_matches(a, f.usage)) {
                flag = &f;
                break;
            }
        }
        if (!flag) return FlagResult::kNotMine;
        uint64_t v = 0;
        ok = take_u64(argc, argv, i, v) && v <= UINT32_MAX;
        cfg.*(flag->field) = static_cast<uint32_t>(v);
    }

    if (!ok) {
        std::fprintf(stderr, "[vf] bad or missing value for %s\n", a);
        return FlagResult::kBad;
    }
    return FlagResult::kOk;
}

bool finish_config(RunConfig& cfg) {
    if (cfg.underlyings.empty()) {
        std::fprintf(stderr, "[vf] --underlying is required\n");
        return false;
    }
    if (cfg.date.empty()) {
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
        localtime_r(&now, &tm);
        char buf[16];
        std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
        cfg.date = buf;
        parse_date(buf, cfg.trade_days);
        std::printf("[vf] no --date given, using today (%s) as the trade date\n", buf);
    }
    if (cfg.queues == 0) {
        std::fprintf(stderr, "[vf] --queues must be at least 1\n");
        return false;
    }
    if (cfg.snapshot_ms == 0) {
        std::fprintf(stderr, "[vf] --snapshot-ms must be at least 1\n");
        return false;
    }
    return true;
}

}  // namespace vf
