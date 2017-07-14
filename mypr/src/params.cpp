#include "params.hpp"
namespace mypr{
std::shared_ptr<CParams> CParams::m_instance = nullptr;

std::shared_ptr<CParams> CParams::instance() {
  if (!m_instance.get()) {
    m_instance = std::make_shared<CParams>(CParams());
  }
  return m_instance;
}
}
