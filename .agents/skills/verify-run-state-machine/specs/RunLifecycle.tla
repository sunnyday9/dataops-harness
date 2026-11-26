--------------------------- MODULE RunLifecycle ---------------------------
EXTENDS Naturals

CONSTANTS
    InitMaxTurns,     \* Initial per-Run turn budget (scaled bound: 2; prod default: 20)
    MaxRejections,    \* HiTL rejection ceiling (policies.yaml: 2)
    ExtTurns,         \* Turn extension per rejection (scaled bound: 2; prod default: 8)
    MaxStrikes        \* Anti-thrash strike ceiling (policies.yaml: 2)

VARIABLES
    status,                 \* One of RunStatus states
    resolution_mode,        \* "fix" | "diagnosis_only"
    turn,                   \* Current turn counter
    max_turns,              \* Effective turn budget ceiling
    rejection_count,        \* HiTL rejection counter (0..MaxRejections)
    thrash_strikes,         \* Anti-thrash consecutive strike counter
    shadow_green,           \* TRUE iff latest ShadowOverlay verification passed
    approval_packet_ready,  \* TRUE iff ApprovalPacket is recorded in AWAITING_APPROVAL
    prod_mutated            \* TRUE iff PromotionService mutated production workspace/storage

vars == <<status, resolution_mode, turn, max_turns, rejection_count,
          thrash_strikes, shadow_green, approval_packet_ready, prod_mutated>>

RunStatuses == {
    "RUNNING",
    "AWAITING_APPROVAL",
    "RESOLVED",
    "DIAGNOSED",
    "ABORTED_THRASH",
    "ABORTED_BUDGET",
    "ABORTED_ERROR"
}

TerminalStatuses == {
    "RESOLVED",
    "DIAGNOSED",
    "ABORTED_THRASH",
    "ABORTED_BUDGET",
    "ABORTED_ERROR"
}

TypeOK ==
    /\ status \in RunStatuses
    /\ resolution_mode \in {"fix", "diagnosis_only"}
    /\ turn \in 0..(InitMaxTurns + MaxRejections * ExtTurns + 1)
    /\ max_turns \in InitMaxTurns..(InitMaxTurns + MaxRejections * ExtTurns)
    /\ rejection_count \in 0..MaxRejections
    /\ thrash_strikes \in 0..MaxStrikes
    /\ shadow_green \in BOOLEAN
    /\ approval_packet_ready \in BOOLEAN
    /\ prod_mutated \in BOOLEAN

Init ==
    /\ status = "RUNNING"
    /\ resolution_mode \in {"fix", "diagnosis_only"}
    /\ turn = 0
    /\ max_turns = InitMaxTurns
    /\ rejection_count = 0
    /\ thrash_strikes = 0
    /\ shadow_green = FALSE
    /\ approval_packet_ready = FALSE
    /\ prod_mutated = FALSE

ToolStepClean ==
    /\ status = "RUNNING"
    /\ turn < max_turns
    /\ turn' = turn + 1
    /\ thrash_strikes' = 0
    /\ shadow_green' = FALSE
    /\ approval_packet_ready' = FALSE
    /\ UNCHANGED <<status, resolution_mode, max_turns, rejection_count, prod_mutated>>

ToolStepThrashStrike ==
    /\ status = "RUNNING"
    /\ turn < max_turns
    /\ thrash_strikes + 1 < MaxStrikes
    /\ turn' = turn + 1
    /\ thrash_strikes' = thrash_strikes + 1
    /\ shadow_green' = FALSE
    /\ approval_packet_ready' = FALSE
    /\ UNCHANGED <<status, resolution_mode, max_turns, rejection_count, prod_mutated>>

AntiThrashAbort ==
    /\ status = "RUNNING"
    /\ turn < max_turns
    /\ thrash_strikes + 1 >= MaxStrikes
    /\ turn' = turn + 1
    /\ thrash_strikes' = thrash_strikes + 1
    /\ status' = "ABORTED_THRASH"
    /\ UNCHANGED <<resolution_mode, max_turns, rejection_count,
                   shadow_green, approval_packet_ready, prod_mutated>>

BudgetAbort ==
    /\ status = "RUNNING"
    /\ turn >= max_turns
    /\ status' = "ABORTED_BUDGET"
    /\ UNCHANGED <<resolution_mode, turn, max_turns, rejection_count,
                   thrash_strikes, shadow_green, approval_packet_ready, prod_mutated>>

UnexpectedErrorAbort ==
    /\ status = "RUNNING"
    /\ status' = "ABORTED_ERROR"
    /\ UNCHANGED <<resolution_mode, turn, max_turns, rejection_count,
                   thrash_strikes, shadow_green, approval_packet_ready, prod_mutated>>

StopDiagnosisOnly ==
    /\ status = "RUNNING"
    /\ turn < max_turns
    /\ resolution_mode = "diagnosis_only"
    /\ turn' = turn + 1
    /\ status' = "DIAGNOSED"
    /\ UNCHANGED <<resolution_mode, max_turns, rejection_count,
                   thrash_strikes, shadow_green, approval_packet_ready, prod_mutated>>

StopFixShadowFail ==
    /\ status = "RUNNING"
    /\ turn < max_turns
    /\ resolution_mode = "fix"
    /\ turn' = turn + 1
    /\ shadow_green' = FALSE
    /\ approval_packet_ready' = FALSE
    /\ UNCHANGED <<status, resolution_mode, max_turns, rejection_count,
                   thrash_strikes, prod_mutated>>

StopFixShadowPass ==
    /\ status = "RUNNING"
    /\ turn < max_turns
    /\ resolution_mode = "fix"
    /\ turn' = turn + 1
    /\ shadow_green' = TRUE
    /\ approval_packet_ready' = TRUE
    /\ status' = "AWAITING_APPROVAL"
    /\ UNCHANGED <<resolution_mode, max_turns, rejection_count,
                   thrash_strikes, prod_mutated>>

CrashResumeCheckpoint ==
    /\ status \in {"RUNNING", "AWAITING_APPROVAL"}
    /\ UNCHANGED vars

HiTLReject ==
    /\ status = "AWAITING_APPROVAL"
    /\ rejection_count < MaxRejections
    /\ rejection_count' = rejection_count + 1
    /\ max_turns' = max_turns + ExtTurns
    /\ shadow_green' = FALSE
    /\ approval_packet_ready' = FALSE
    /\ status' = "RUNNING"
    /\ UNCHANGED <<resolution_mode, turn, thrash_strikes, prod_mutated>>

HiTLApprovePromote ==
    /\ status = "AWAITING_APPROVAL"
    /\ resolution_mode = "fix"
    /\ shadow_green = TRUE
    /\ approval_packet_ready = TRUE
    /\ prod_mutated' = TRUE
    /\ status' = "RESOLVED"
    /\ UNCHANGED <<resolution_mode, turn, max_turns, rejection_count,
                   thrash_strikes, shadow_green, approval_packet_ready>>

TerminalStutter ==
    /\ status \in TerminalStatuses
    /\ UNCHANGED vars

Next ==
    \/ ToolStepClean
    \/ ToolStepThrashStrike
    \/ AntiThrashAbort
    \/ BudgetAbort
    \/ UnexpectedErrorAbort
    \/ StopDiagnosisOnly
    \/ StopFixShadowFail
    \/ StopFixShadowPass
    \/ CrashResumeCheckpoint
    \/ HiTLReject
    \/ HiTLApprovePromote
    \/ TerminalStutter

Spec == Init /\ [][Next]_vars

\* Core Safety Invariants
NoUnapprovedProdMutation ==
    prod_mutated => (status = "RESOLVED")

PromotionRequiresGreenShadow ==
    (status = "RESOLVED") => (shadow_green /\ approval_packet_ready /\ resolution_mode = "fix")

AwaitingApprovalRequiresGreenShadow ==
    (status = "AWAITING_APPROVAL") => (shadow_green /\ approval_packet_ready /\ resolution_mode = "fix")

DiagnosisOnlyNeverPromotes ==
    (resolution_mode = "diagnosis_only") =>
        (~prod_mutated /\ status /= "AWAITING_APPROVAL" /\ status /= "RESOLVED")

BoundedRejections ==
    rejection_count <= MaxRejections

TurnBudgetBounded ==
    /\ max_turns <= InitMaxTurns + MaxRejections * ExtTurns
    /\ (status = "RUNNING" => turn <= max_turns)

=============================================================================
