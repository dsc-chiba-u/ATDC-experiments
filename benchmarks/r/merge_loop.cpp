// Union-find loop of the merge tree in C++ (Rcpp), same output as merge_loop_r.
#include <Rcpp.h>
#include <set>
using namespace Rcpp;

static inline int find(std::vector<int>& p, int x) {
  while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
  return x;
}

// [[Rcpp::export]]
List merge_loop_cpp(NumericVector f, IntegerVector eu, IntegerVector ev, NumericVector w) {
  int n = f.size(), ne = eu.size();
  std::vector<int> p(n), peak(n);
  for (int i = 0; i < n; ++i) { p[i] = i; peak[i] = i; }
  std::vector<int> dying, surviving, mu, mv;
  std::vector<double> level;
  for (int e = 0; e < ne; ++e) {
    int a = find(p, eu[e] - 1), b = find(p, ev[e] - 1);
    if (a == b) continue;
    int pa = peak[a], pb = peak[b];
    if (f[pa] < f[pb] || (f[pa] == f[pb] && pa > pb)) { std::swap(a, b); std::swap(pa, pb); }
    p[b] = a;
    dying.push_back(pb + 1); surviving.push_back(pa + 1); level.push_back(w[e]);
    mu.push_back(eu[e]); mv.push_back(ev[e]);
  }
  std::set<int> roots;
  for (int i = 0; i < n; ++i) roots.insert(peak[find(p, i)] + 1);
  return List::create(_["dying"] = dying, _["surviving"] = surviving, _["level"] = level,
                      _["mu"] = mu, _["mv"] = mv, _["roots"] = IntegerVector(roots.begin(), roots.end()));
}
