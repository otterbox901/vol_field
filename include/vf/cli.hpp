#pragma once
//
// vf/cli.hpp - command-line flags shared by the driver binaries.
//

#include <cstdint>
#include <string>
#include <vector>

#include "vf/config.hpp"

namespace vf {

/// Help text lines up at this column, in every binary's usage.
constexpr int kUsageWidth = 22;

/// Print the help lines for the flags parse_common_flag() accepts.
void print_common_usage();

/// Parse the value after argv[i], advancing i. False if missing or malformed.
bool take_u64(int argc, char** argv, int& i, uint64_t& out);
bool take_double(int argc, char** argv, int& i, double& out);
bool take_str(int argc, char** argv, int& i, const char*& out);

/// "YYYY-MM-DD" -> days since 1970-01-01.
bool parse_date(const char* s, int32_t& days);

/// A UDP destination: a multicast group (or unicast address) and port.
struct Endpoint {
    uint32_t ip;    ///< host order; 0 = any
    uint16_t port;
};

/// "A.B.C.D:PORT", ":PORT" or "PORT" -> address (host order, 0 = any) and port.
bool parse_endpoint(const char* s, uint32_t& ip, uint16_t& port);

/// "G:P,G:P,..." appended to `out`. False on a malformed or empty list.
bool parse_endpoints(const char* s, std::vector<Endpoint>& out);

/// "A.B.C.D:PORT" (or ":PORT" when ip is 0), for messages.
std::string endpoint_string(const Endpoint& e);

/// Split "a,b,c" into its non-empty parts.
std::vector<std::string> split_list(const char* s);

enum class FlagResult { kNotMine, kOk, kBad, kHelp };

/// Try argv[i] as one of the common flags. On kOk, i has been advanced past
/// any value.
FlagResult parse_common_flag(int argc, char** argv, int& i, RunConfig& cfg);

/// Apply defaults (today's date) and check what the flags left. Prints why
/// and returns false when the configuration cannot run.
bool finish_config(RunConfig& cfg);

}  // namespace vf
