//
// tests/test_black.cpp - Black-76, the implied-vol solver, time to expiry and
// the parity forward, on the host. The kernels run the same VF_HD code.
//

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "vf/black.hpp"
#include "vf/common.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                        \
    } while (0)

/// Price -> IV round trip over K/F in [0.5, 2], T in [1 day, 2 years], sigma in
/// [0.05, 2]. The out-of-the-money side (what the surface uses) must come back
/// within 1e-7 wherever vega > 1e-8. The in-the-money side goes through parity,
/// where the price itself only carries ~1e-16 * intrinsic of time value, so it
/// is held to the same 1e-7 only where vega > 1e-5.
void test_round_trip() {
    const double F = 100.0;
    const double r = 0.04;
    const double days[] = {1, 2, 7, 30, 91, 182, 365, 730};
    const double vols[] = {0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 1.6, 2.0};

    int otm_checked = 0, otm_skipped = 0, itm_checked = 0;
    double worst = 0.0;
    for (double d : days) {
        const double T = d / 365.0;
        const double df = std::exp(-r * T);
        for (int i = 0; i <= 60; ++i) {
            const double K = F * std::exp(std::log(0.5) + i * (std::log(4.0) / 60.0));
            for (double sigma : vols) {
                const double vega = vf::black76_vega(F, K, T, sigma, df);
                for (int side = 0; side < 2; ++side) {
                    const bool call = side == 0;
                    const bool otm = call ? K >= F : K < F;
                    if (vega <= (otm ? 1e-8 : 1e-5)) {
                        if (otm) ++otm_skipped;
                        continue;
                    }
                    const double price = vf::black76_price(F, K, T, sigma, df, call);
                    const double iv = vf::implied_vol(price, F, K, T, df, call);
                    const double err = std::fabs(iv - sigma);
                    if (!(err < 1e-7)) {
                        ++g_failures;
                        std::fprintf(stderr,
                                     "FAIL round trip: K/F=%.4f T=%gd sigma=%g %s vega=%.3g "
                                     "-> iv=%.12g\n",
                                     K / F, d, sigma, call ? "call" : "put", vega, iv);
                    }
                    ++g_checks;
                    if (err > worst) worst = err;
                    if (otm) ++otm_checked;
                    else ++itm_checked;
                }
            }
        }
    }
    std::printf("  round trip: %d OTM + %d ITM points (OTM skipped for vega <= 1e-8: %d), "
                "worst |dsigma| = %.2e\n",
                otm_checked, itm_checked, otm_skipped, worst);
}

/// Prices the solver must reject.
void test_bad_prices() {
    const double F = 100.0, T = 0.25, df = std::exp(-0.04 * T);
    auto nan = [](double v) { return std::isnan(v); };

    CHECK(nan(vf::implied_vol(0.0, F, 100, T, df, true)));
    CHECK(nan(vf::implied_vol(-1.0, F, 100, T, df, true)));
    CHECK(nan(vf::implied_vol(NAN, F, 100, T, df, true)));
    // Below intrinsic: an ITM call worth less than df (F - K).
    CHECK(nan(vf::implied_vol(df * 9.99, F, 90, T, df, true)));
    CHECK(nan(vf::implied_vol(df * 10.0, F, 90, T, df, true)));  // exactly intrinsic
    // At or above the cap: a call is worth less than df F, a put less than df K.
    CHECK(nan(vf::implied_vol(df * F, F, 100, T, df, true)));
    CHECK(nan(vf::implied_vol(df * 120.0, F, 120, T, df, false)));
    // Degenerate inputs.
    CHECK(nan(vf::implied_vol(2.0, F, 100, 0.0, df, true)));
    CHECK(nan(vf::implied_vol(2.0, 0.0, 100, T, df, true)));
    CHECK(nan(vf::implied_vol(2.0, F, 0.0, T, df, true)));
    CHECK(nan(vf::implied_vol(2.0, NAN, 100, T, df, true)));
    // A sane price is accepted.
    CHECK(!nan(vf::implied_vol(4.0, F, 100, T, df, true)));
}

void test_time() {
    CHECK(vf::days_from_civil(1970, 1, 1) == 0);
    CHECK(vf::days_from_civil(2000, 1, 1) == 10957);
    CHECK(vf::days_from_civil(2024, 2, 29) == 19782);
    CHECK(vf::days_from_civil(2026, 10, 20) == 20746);

    // 9:30 on the trade date to 16:00 the next day.
    const double t = vf::time_to_expiry(1, 34200);
    CHECK(std::fabs(t - (86400.0 + 57600.0 - 34200.0) / (365.0 * 86400.0)) < 1e-15);
    // Expiry day after the close, and an expired date: floored at one minute.
    CHECK(vf::time_to_expiry(0, 16 * 3600 + 5) == 60.0 / (365.0 * 86400.0));
    CHECK(vf::time_to_expiry(-3, 34200) == 60.0 / (365.0 * 86400.0));
}

/// Parity forward on a synthetic chain priced off a known forward.
void test_forward() {
    const double F = 101.3, r = 0.04, T = 0.25;
    const double df = std::exp(-r * T), growth = std::exp(r * T);
    std::vector<double> K, cmp;
    for (double k = 80.0; k <= 120.0 + 1e-9; k += 2.5) {
        const double sigma = 0.22 - 0.1 * std::log(k / F);  // a skew; parity does not care
        K.push_back(k);
        cmp.push_back(vf::black76_price(F, k, T, sigma, df, true) -
                      vf::black76_price(F, k, T, sigma, df, false));
    }
    const uint32_t n = static_cast<uint32_t>(K.size());

    uint32_t used = 0;
    CHECK(std::fabs(vf::parity_forward(K.data(), cmp.data(), n, growth, &used) - F) < 1e-9);
    CHECK(used == vf::kForwardPairs);

    // Noisy mids (+-half a penny on each leg): the median stays within a cent.
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> noise(-0.01, 0.01);
    std::vector<double> noisy = cmp;
    for (double& v : noisy) v += noise(rng);
    CHECK(std::fabs(vf::parity_forward(K.data(), noisy.data(), n, growth) - F) < 0.011);

    // One wild value among the voters is outvoted; wild values far from the
    // money never vote at all.
    std::vector<double> wild = cmp;
    size_t atm = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::fabs(cmp[i]) < std::fabs(cmp[atm])) atm = i;
    wild[atm] += 0.5;
    wild[0] += 50.0;
    wild[n - 1] -= 50.0;
    CHECK(std::fabs(vf::parity_forward(K.data(), wild.data(), n, growth) - F) < 1e-6);

    // NaN pairs are skipped; nothing usable gives NaN; an even count averages.
    std::vector<double> holes = cmp;
    holes[atm] = NAN;
    CHECK(std::fabs(vf::parity_forward(K.data(), holes.data(), n, growth) - F) < 1e-9);
    CHECK(std::isnan(vf::parity_forward(K.data(), cmp.data(), 0, growth, &used)) && used == 0);
    const double k2[] = {100.0, 102.0};
    const double c2[] = {1.0, -1.0};  // votes 101 and 101, at growth 1
    CHECK(vf::parity_forward(k2, c2, 2, 1.0, &used) == 101.0 && used == 2);
}

}  // namespace

int main() {
    test_round_trip();
    test_bad_prices();
    test_time();
    test_forward();
    std::printf("test_black: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
