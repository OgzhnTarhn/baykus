// CTest bu testi iki ortamla koşar (CMakeLists.txt): BAYKUS_REQUIRE_FIXTURES=1 iken başarısız
// olmalı (WILL_FAIL), 0 iken atlanmalı (başarılı sayılır).
#include "fixtures.hpp"

TEST(FixtureProbe, IndexPresent) {
  BAYKUS_REQUIRE_FILES({baykus::test::fixtures_root() / "index.json"});
  SUCCEED();
}
