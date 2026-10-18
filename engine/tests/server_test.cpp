// HTTP API (Kabul 8): uç noktalar, hata kodları ve 1000 rastgele istekten sonra /health. Sunucu
// süreç içinde, rastgele portta; istemci httplib::Client. BAYKUS_EVIDENCE_DIR verilirse kanıt
// `api.json` oraya yazılır (results/system/api.json).
#include <gtest/gtest.h>
#include <httplib.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include "baykus/models.hpp"
#include "baykus/runtime.hpp"
#include "baykus/server.hpp"
#include "baykus/sessions.hpp"
#include "fixtures.hpp"

namespace bt = baykus::test;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

fs::path configs() {
  const char* c = std::getenv("BAYKUS_CONFIGS");
  return fs::path(c && *c ? c : "/configs") / "engine";
}

fs::path clip_file(const std::string& sub, int frame) {
  char name[16];
  std::snprintf(name, sizeof(name), "%06d", frame);
  return bt::fixtures_root() / "m3ot" / "clip" / "sequences" / "clip" / sub /
         (std::string(name) + (sub == "rgb" ? ".jpg" : ".png"));
}

httplib::UploadFormDataItems parts(const std::string& rgb, const std::string& ir) {
  httplib::UploadFormDataItems items;
  if (!rgb.empty()) items.push_back({"rgb", rgb, "rgb.jpg", "image/jpeg"});
  if (!ir.empty()) items.push_back({"ir", ir, "ir.png", "image/png"});
  return items;
}

}  // namespace

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto m = bt::models_root();
    BAYKUS_REQUIRE_FILES({clip_file("rgb", 1), m / "e3c.onnx", configs() / "models.json",
                          configs() / "tracker.json"});
    models_ = std::make_unique<baykus::Models>(
        baykus::load_models(m.string(), (configs() / "models.json").string()));
    cfg_ = baykus::TrackerConfig::load((configs() / "tracker.json").string());
    sessions_ = std::make_unique<baykus::Sessions>(cfg_, models_->gate.window,
                                                   models_->gate.min_frames, models_->gate.tol_px);
    runtime_ = std::make_unique<baykus::Runtime>(*models_, cfg_, *sessions_);
    server_ = std::make_unique<baykus::Server>(*models_, cfg_, *runtime_, *sessions_);
    port_ = server_->bind_any_port("127.0.0.1");
    ASSERT_GT(port_, 0);
    thread_ = std::thread([this] { server_->listen_after_bind(); });
    server_->wait_until_ready();
    client_ = std::make_unique<httplib::Client>("127.0.0.1", port_);
    client_->set_read_timeout(60, 0);
    rgb_ = bt::read_file(clip_file("rgb", 1));
    ir_ = bt::read_file(clip_file("ir", 1));
  }

  void TearDown() override {
    if (server_) server_->stop();
    if (thread_.joinable()) thread_.join();
  }

  int post_detect(const std::string& query, const httplib::UploadFormDataItems& items) {
    const auto r = client_->Post("/v1/detect" + query, items);
    return r ? r->status : -1;
  }

  // Kabul 8 vakaları: (ad, beklenen durum, gerçek durum)
  json cases() {
    const std::string tiny =
        bt::read_file(bt::fixtures_root() / "synthetic" / "tiny_33x17_rgb.png");
    const std::string corrupt = ir_.substr(0, ir_.size() / 2);
    const std::string huge(baykus::kMaxPartBytes + 1, '\x01');
    const std::vector<std::tuple<std::string, int, int>> list = {
        {"detect_ok", 200, post_detect("?track=true&session=k8", parts(rgb_, ir_))},
        {"missing_part", 400, post_detect("", parts(rgb_, ""))},
        {"size_mismatch", 400, post_detect("", parts(tiny, ir_))},
        {"corrupt_png", 400, post_detect("", parts(rgb_, corrupt))},
        {"unknown_gate", 400, post_detect("?gate=bogus", parts(rgb_, ir_))},
        {"conf_out_of_range", 400, post_detect("?conf=1.5", parts(rgb_, ir_))},
        {"conf_not_a_number", 400, post_detect("?conf=abc", parts(rgb_, ir_))},
        {"bad_track_flag", 400, post_detect("?track=maybe", parts(rgb_, ir_))},
        {"track_without_session", 400, post_detect("?track=true", parts(rgb_, ir_))},
        {"part_too_large", 413, post_detect("", parts(huge, ir_))},
        {"not_multipart", 400,
         [&] {
           const auto r = client_->Post("/v1/detect", "xyz", "application/octet-stream");
           return r ? r->status : -1;
         }()},
        {"reset_without_session", 400,
         [&] {
           const auto r = client_->Post("/v1/session/reset", "", "text/plain");
           return r ? r->status : -1;
         }()},
        {"reset_ok", 200,
         [&] {
           const auto r = client_->Post("/v1/session/reset?session=k8", "", "text/plain");
           return r ? r->status : -1;
         }()},
        {"unknown_path", 404,
         [&] {
           const auto r = client_->Get("/v1/yok");
           return r ? r->status : -1;
         }()},
    };
    json out = json::array();
    for (const auto& [name, want, got] : list) {
      out.push_back({{"name", name}, {"expected", want}, {"status", got}, {"ok", want == got}});
    }
    return out;
  }

  // 1000 rastgele istek (seed sabit): rastgele baytlı parçalar, bozuk gövdeler, rastgele yollar.
  json fuzz(int n = 1000) {
    std::mt19937 rng(42);
    auto bytes = [&](std::size_t max) {
      std::string s(std::uniform_int_distribution<std::size_t>(0, max)(rng), '\0');
      for (auto& c : s) c = static_cast<char>(rng() & 0xff);
      return s;
    };
    std::map<int, int> statuses;
    for (int i = 0; i < n; ++i) {
      httplib::Result r;
      switch (rng() % 5) {
        case 0:
          r = client_->Post("/v1/detect", parts(bytes(4096), bytes(4096)));
          break;
        case 1:  // gerçek RGB, rastgele IR
          r = client_->Post("/v1/detect?track=true&session=fz", parts(rgb_, bytes(2048)));
          break;
        case 2:
          r = client_->Post("/v1/detect", bytes(8192), "multipart/form-data; boundary=x");
          break;
        case 3:
          r = client_->Post("/v1/detect?conf=" + std::to_string(rng() % 300 / 100.0), bytes(512),
                            "application/octet-stream");
          break;
        default:
          r = client_->Get("/" + std::to_string(rng()));
      }
      ++statuses[r ? r->status : -1];
    }
    const auto h = client_->Get("/health");
    const bool healthy = h && h->status == 200 && json::parse(h->body).at("status") == "ok";
    json hist = json::object();
    int server_errors = 0;
    for (const auto& [code, count] : statuses) {
      hist[std::to_string(code)] = count;
      if (code < 0 || code >= 500) server_errors += count;
    }
    return {{"n", n},
            {"seed", 42},
            {"statuses", hist},
            {"server_errors", server_errors},
            {"healthy_after", healthy}};
  }

  std::unique_ptr<baykus::Models> models_;
  baykus::TrackerConfig cfg_;
  std::unique_ptr<baykus::Sessions> sessions_;
  std::unique_ptr<baykus::Runtime> runtime_;
  std::unique_ptr<baykus::Server> server_;
  std::unique_ptr<httplib::Client> client_;
  std::thread thread_;
  int port_ = -1;
  std::string rgb_, ir_;
};

TEST_F(ServerTest, HealthOk) {
  const auto r = client_->Get("/health");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 200);
  EXPECT_EQ(json::parse(r->body), json::parse(R"({"status": "ok", "model_loaded": true})"));
}

TEST_F(ServerTest, ModelShowsShas) {
  const auto r = client_->Get("/v1/model");
  ASSERT_TRUE(r);
  ASSERT_EQ(r->status, 200);
  const auto j = json::parse(r->body);
  const auto& files = models_->manifest.at("files");
  EXPECT_EQ(j.at("sha256").at("model"), files.at("e3c.onnx").at("sha256"));
  EXPECT_EQ(j.at("sha256").at("thresholds"), files.at("thresholds.json").at("sha256"));
  EXPECT_EQ(j.at("sha256").at("calibration"), files.at("calibration.json").at("sha256"));
  EXPECT_EQ(j.at("sha256").at("gate_model"), files.at("gate_model.json").at("sha256"));
  EXPECT_EQ(j.at("tracker"), cfg_.to_json());
  EXPECT_EQ(j.at("classes"), json::parse(R"(["car", "truck", "bus", "van", "freight_car"])"));
  EXPECT_EQ(j.at("input_size"), json::parse("[640, 640]"));
  EXPECT_EQ(j.at("channels"), 4);
  EXPECT_EQ(j.at("mode"), "early_fusion");
  EXPECT_EQ(j.at("precision"), "fp32");
}

TEST_F(ServerTest, DetectShape) {
  const auto r =
      client_->Post("/v1/detect?conf=0.25&track=true&session=a&gate=auto", parts(rgb_, ir_));
  ASSERT_TRUE(r);
  ASSERT_EQ(r->status, 200) << r->body;
  const auto j = json::parse(r->body);
  EXPECT_EQ(j.at("session_new"), true);
  ASSERT_FALSE(j.at("detections").empty());
  for (const auto& d : j.at("detections")) {
    for (const char* k : {"cls", "cls_id", "score", "score_raw", "box", "track_id"}) {
      EXPECT_TRUE(d.contains(k)) << k;
    }
    EXPECT_GE(d.at("score").get<double>(), 0.25);
    EXPECT_TRUE(d.at("track_id").is_null());  // ilk karede olgun iz yok (referans -1)
    for (const char* k : {"x1", "y1", "x2", "y2"}) {
      EXPECT_TRUE(d.at("box").contains(k)) << k;
    }
  }
  for (const char* k : {"decode", "health", "pre", "infer", "post", "track", "total"}) {
    EXPECT_TRUE(j.at("timing_ms").contains(k)) << k;
  }
  EXPECT_EQ(j.at("timing_ms").size(), 7u);
  for (const char* k : {"rgb", "ir", "pair", "gate", "reason", "reason_code", "applied"}) {
    EXPECT_TRUE(j.at("health").contains(k)) << k;
  }
  // İkinci kare aynı oturumda: oturum yeni değil, olgun izler kimlik alır
  const auto r2 = client_->Post("/v1/detect?track=true&session=a", parts(rgb_, ir_));
  ASSERT_TRUE(r2);
  const auto j2 = json::parse(r2->body);
  EXPECT_EQ(j2.at("session_new"), false);
  int with_id = 0;
  for (const auto& d : j2.at("detections")) with_id += d.at("track_id").is_number() ? 1 : 0;
  EXPECT_GT(with_id, 0);
}

TEST_F(ServerTest, Errors400) {
  for (const auto& c : cases()) {
    if (c.at("expected") == 400) {
      EXPECT_TRUE(c.at("ok").get<bool>()) << c.dump();
    }
  }
}

TEST_F(ServerTest, Error413) {
  const auto r =
      client_->Post("/v1/detect", parts(std::string(baykus::kMaxPartBytes + 1, '\x01'), ir_));
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 413);
  EXPECT_TRUE(json::parse(r->body).contains("error"));
}

TEST_F(ServerTest, ErrorsAreJson) {
  for (const std::string path : {"/v1/yok", "/v1/session/reset"}) {
    const auto r = path == "/v1/yok" ? client_->Get(path) : client_->Post(path, "", "text/plain");
    ASSERT_TRUE(r) << path;
    EXPECT_TRUE(json::parse(r->body).contains("error")) << path << ": " << r->body;
  }
}

TEST_F(ServerTest, FuzzThenHealthy) {
  const auto f = fuzz();
  EXPECT_TRUE(f.at("healthy_after").get<bool>()) << f.dump();
  EXPECT_EQ(f.at("server_errors"), 0) << f.dump();
}

TEST_F(ServerTest, WritesApiEvidence) {
  const json rec = {{"kabul", "8"}, {"cases", cases()}, {"fuzz", fuzz()}};
  for (const auto& c : rec.at("cases")) {
    EXPECT_TRUE(c.at("ok").get<bool>()) << c.dump();
  }
  EXPECT_TRUE(rec.at("fuzz").at("healthy_after").get<bool>());
  const char* dir = std::getenv("BAYKUS_EVIDENCE_DIR");
  if (dir != nullptr && *dir != '\0')
    std::ofstream(fs::path(dir) / "api.json") << rec.dump(2) << "\n";
}
