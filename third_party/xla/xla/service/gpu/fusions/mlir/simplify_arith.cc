/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/passes.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_SIMPLIFYARITHPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

// Returns the range of a given value, if it can be statically determined.
std::optional<Interval> GetRange(mlir::Value value) {
  auto attr_to_range = [](mlir::Attribute attr) -> std::optional<Interval> {
    if (!attr) {
      return std::nullopt;
    }
    auto values = llvm::to_vector(
        mlir::cast<mlir::ArrayAttr>(attr).getAsValueRange<mlir::IntegerAttr>());
    return {{values[0].getSExtValue(), values[1].getSExtValue()}};
  };

  if (value.getDefiningOp()) {
    return attr_to_range(value.getDefiningOp()->getAttr("xla.range"));
  }

  auto bbarg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!bbarg) {
    return std::nullopt;
  }

  auto parent = bbarg.getParentBlock()->getParentOp();
  if (auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(parent)) {
    return attr_to_range(func_op.getArgAttr(bbarg.getArgNumber(), "xla.range"));
  }

  if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
    llvm::APInt lb, ub;
    if (mlir::matchPattern(for_op.getLowerBound(), mlir::m_ConstantInt(&lb)) &&
        mlir::matchPattern(for_op.getUpperBound(), mlir::m_ConstantInt(&ub))) {
      return {{lb.getSExtValue(), ub.getSExtValue() - 1}};
    }
  }
  return std::nullopt;
}

Interval::ComparisonResult EvaluateCmpI(mlir::arith::CmpIPredicate pred,
                                        Interval lhs, int64_t rhs) {
  switch (pred) {
    case mlir::arith::CmpIPredicate::eq:
      return lhs == rhs;
    case mlir::arith::CmpIPredicate::ne:
      return lhs != rhs;
    case mlir::arith::CmpIPredicate::slt:
    case mlir::arith::CmpIPredicate::ult:
      return lhs < rhs;
    case mlir::arith::CmpIPredicate::sle:
    case mlir::arith::CmpIPredicate::ule:
      return lhs <= rhs;
    case mlir::arith::CmpIPredicate::sgt:
    case mlir::arith::CmpIPredicate::ugt:
      return lhs > rhs;
    case mlir::arith::CmpIPredicate::sge:
    case mlir::arith::CmpIPredicate::uge:
      return lhs >= rhs;
  }
}

struct RewriteCmpI : mlir::OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::CmpIOp op, mlir::PatternRewriter& rewriter) const override {
    // We don't need to support constants on the LHS, since comparisons are
    // canonicalized to have them on the RHS.
    auto rhs = mlir::getConstantIntValue(op.getRhs());
    auto lhs = GetRange(op.getLhs());
    if (lhs && rhs) {
      Interval::ComparisonResult result =
          EvaluateCmpI(op.getPredicate(), *lhs, *rhs);
      if (result != std::nullopt) {
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(
            op, *result, rewriter.getI1Type());
        return mlir::success();
      }
    }
    // TODO(jreiffers): Consider supporting ranges on the RHS as well.
    return rewriter.notifyMatchFailure(op, "not a constant result");
  }
};

class SimplifyArithPass
    : public impl::SimplifyArithPassBase<SimplifyArithPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteCmpI>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateSimplifyArithPass() {
  return std::make_unique<SimplifyArithPass>();
}

}  // namespace gpu
}  // namespace xla
