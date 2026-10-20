#pragma once
//
// vf/black.hpp - Black-76 on the forward, implied volatility, time to expiry
// and the put-call parity forward.
//
// Double precision and VF_HD throughout: the receive and surface kernels run
// exactly the code tests/test_black.cpp checks on the host.
//
// Conventions
//   * Prices are discounted option premiums; df = e^{-rT}. The rate is a
//     constant (--rate); dividends and borrow are absorbed by the forward,
//     which comes from parity rather than from spot.
//   * T is in years of 365 days, measured to 16:00 ET on the expiry date and
//     floored at one minute.
//   * Nasdaq options are American. A European model on American prices is
//     biased in the money, so the surface only uses out-of-the-money prices.
//

#include <cmath>

#include "vf/common.hpp"

namespace vf {

constexpr double kSecondsPerYear = 365.0 * 86400.0;
constexpr double kExpiryCloseSec = 16.0 * 3600.0;  ///< expiry is at 16:00 on the expiry date
constexpr double kMinTimeSec = 60.0;               ///< floor on time to expiry
constexpr double kMaxVol = 10.0;                   ///< bracket top: 1000% volatility
constexpr int kMaxIvIterations = 100;
constexpr uint32_t kForwardPairs = 5;              ///< pairs that vote on the forward

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrt2Pi = 2.50662827463100050242;
constexpr double kInvSqrt2Pi = 0.39894228040143267794;
constexpr double kSqrtHalf = 0.70710678118654752440;

VF_HD VF_FORCEINLINE double quiet_nan() { return static_cast<double>(NAN); }

VF_HD VF_FORCEINLINE double norm_pdf(double x) { return kInvSqrt2Pi * exp(-0.5 * x * x); }

/// Standard normal CDF through erfc, which keeps full relative precision deep
/// in the lower tail - where out-of-the-money prices live.
VF_HD VF_FORCEINLINE double norm_cdf(double x) { return 0.5 * erfc(-x * kSqrtHalf); }

/// Undiscounted Black-76 price.
VF_HD inline double black76_undiscounted(double F, double K, double T, double sigma, bool call) {
    const double sd = sigma * sqrt(T);
    const double d1 = log(F / K) / sd + 0.5 * sd;
    const double d2 = d1 - sd;
    return call ? F * norm_cdf(d1) - K * norm_cdf(d2) : K * norm_cdf(-d2) - F * norm_cdf(-d1);
}

/// Discounted Black-76 price.
VF_HD inline double black76_price(double F, double K, double T, double sigma, double df,
                                  bool call) {
    return df * black76_undiscounted(F, K, T, sigma, call);
}

/// dPrice/dSigma of the discounted price (same for calls and puts).
VF_HD inline double black76_vega(double F, double K, double T, double sigma, double df) {
    const double sqrt_t = sqrt(T);
    const double sd = sigma * sqrt_t;
    const double d1 = log(F / K) / sd + 0.5 * sd;
    return df * F * norm_pdf(d1) * sqrt_t;
}

/// Implied volatility of a discounted Black-76 price.
///
/// The solve runs on the out-of-the-money side: an in-the-money price is
/// turned into its time value through parity (C - P = df (F - K)), which is
/// the OTM option's price at the same strike and has no intrinsic part to
/// cancel against. It starts from Corrado-Miller (Brenner-Subrahmanyam when
/// that is not usable), then takes Newton steps inside a bracket that every
/// evaluation tightens; a step that would leave the bracket, or a vanishing
/// vega, bisects instead, so it always converges.
///
/// Returns NaN for non-positive inputs, and for a price outside the
/// no-arbitrage bounds: at or below intrinsic, or at or above the cap (F for a
/// call, K for a put, undiscounted) - or beyond what kMaxVol explains.
VF_HD inline double implied_vol(double price, double F, double K, double T, double df,
                                bool call) {
    if (!(price > 0.0) || !(F > 0.0) || !(K > 0.0) || !(T > 0.0) || !(df > 0.0))
        return quiet_nan();

    const double p = price / df;
    const double intrinsic = call ? fmax(F - K, 0.0) : fmax(K - F, 0.0);
    const double tv = p - intrinsic;  // = undiscounted price of the OTM option at K
    const bool otm_call = K >= F;
    if (!(tv > 0.0) || !(tv < (otm_call ? F : K))) return quiet_nan();
    if (!(black76_undiscounted(F, K, T, kMaxVol, otm_call) > tv)) return quiet_nan();

    const double sqrt_t = sqrt(T);
    const double log_fk = log(F / K);

    // Corrado-Miller, written for the call price at this strike.
    const double c = otm_call ? tv : tv + (F - K);
    const double x = c - 0.5 * (F - K);
    const double rad = x * x - (F - K) * (F - K) / kPi;
    double sigma = kSqrt2Pi / (F + K) * (x + sqrt(fmax(rad, 0.0))) / sqrt_t;
    // Fall back to Brenner-Subrahmanyam, then to a flat guess.
    if (!(sigma > 0.0 && sigma < kMaxVol)) sigma = kSqrt2Pi * tv / F / sqrt_t;
    if (!(sigma > 0.0 && sigma < kMaxVol)) sigma = 0.5;

    double lo = 0.0;
    double hi = kMaxVol;
    for (int it = 0; it < kMaxIvIterations; ++it) {
        const double sd = sigma * sqrt_t;
        const double d1 = log_fk / sd + 0.5 * sd;
        const double d2 = d1 - sd;
        const double model = otm_call ? F * norm_cdf(d1) - K * norm_cdf(d2)
                                      : K * norm_cdf(-d2) - F * norm_cdf(-d1);
        const double diff = model - tv;
        if (diff == 0.0) return sigma;
        if (diff > 0.0) hi = sigma;
        else lo = sigma;

        double next = sigma - diff / (F * norm_pdf(d1) * sqrt_t);
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);  // also catches vega == 0
        if (fabs(next - sigma) <= 1e-13 * sigma || hi - lo <= 1e-14 * hi) return next;
        sigma = next;
    }
    return hi - lo <= 1e-9 * hi ? 0.5 * (lo + hi) : quiet_nan();
}

/// Years from `feed_sec` (seconds past midnight on the trade date) to 16:00 on
/// an expiry `days_to_expiry` days after the trade date, floored at a minute.
VF_HD inline double time_to_expiry(int32_t days_to_expiry, uint32_t feed_sec) {
    const double sec = static_cast<double>(days_to_expiry) * 86400.0 + kExpiryCloseSec -
                       static_cast<double>(feed_sec);
    return fmax(sec, kMinTimeSec) / kSecondsPerYear;
}

/// Forward implied by put-call parity across one expiry's strikes.
///
/// Each pair with both mids gives F_i = K_i + growth (C_i - P_i), with growth =
/// e^{rT}. Pairs far from the money carry the most early-exercise premium and
/// the widest relative spreads, so only the kForwardPairs pairs with the
/// smallest |C - P| vote, and the median of their F_i is the answer. NaN
/// entries are skipped. Returns NaN when nothing votes; `used` (optional)
/// receives the number of voters.
VF_HD inline double parity_forward(const double* strike, const double* call_minus_put,
                                   uint32_t n, double growth, uint32_t* used = nullptr) {
    uint32_t best[kForwardPairs];
    uint32_t nb = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const double d = fabs(call_minus_put[i]);
        if (!(d == d) || !(strike[i] > 0.0)) continue;
        if (nb == kForwardPairs && d >= fabs(call_minus_put[best[nb - 1]])) continue;
        uint32_t j = nb < kForwardPairs ? nb++ : kForwardPairs - 1;
        while (j > 0 && fabs(call_minus_put[best[j - 1]]) > d) {
            best[j] = best[j - 1];
            --j;
        }
        best[j] = i;
    }
    if (used) *used = nb;
    if (nb == 0) return quiet_nan();

    double f[kForwardPairs];
    for (uint32_t k = 0; k < nb; ++k) {
        const double v = strike[best[k]] + growth * call_minus_put[best[k]];
        uint32_t j = k;
        while (j > 0 && f[j - 1] > v) {
            f[j] = f[j - 1];
            --j;
        }
        f[j] = v;
    }
    return (nb & 1u) ? f[nb / 2] : 0.5 * (f[nb / 2 - 1] + f[nb / 2]);
}

}  // namespace vf
