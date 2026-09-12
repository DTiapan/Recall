# Enterprise Information Security Policy

**Document ID:** ISP-2026-04  
**Effective Date:** 2026-04-01  
**Owner:** Chief Information Security Officer (CISO)

## 1. Purpose

This policy establishes minimum security controls for all employees, contractors, and third-party systems that access Recall Enterprise data.

## 2. Multi-Factor Authentication (MFA)

All workforce accounts with access to production systems **must** enroll in MFA within 14 days of onboarding. Approved factors include FIDO2 hardware keys, TOTP authenticator apps, and corporate SSO push notifications. Shared credentials are prohibited.

## 3. Supply Chain Integrity

Following the discovery of **CVE-2024-3094** in upstream **xz Utils** (liblzma backdoor), all build pipelines must verify package checksums and run Software Bill of Materials (SBOM) scans before deployment. Any dependency flagged as compromised must be quarantined within 4 hours.

## 4. Incident Response

Security incidents are classified P1–P4. P1 events (active data exfiltration, ransomware) require immediate escalation to the Security Operations Center via the `#sec-incident` Slack channel and activation of the 24/7 on-call roster.

## 5. Data Classification

| Level | Examples | Storage |
|-------|----------|---------|
| Public | Marketing site copy | Any approved CDN |
| Internal | Architecture diagrams | Encrypted object storage |
| Confidential | Customer PII, financial forecasts | AES-256 at rest, TLS 1.3 in transit |

## 6. Review Cycle

This policy is reviewed quarterly. The next scheduled review is **2026-07-01**.
