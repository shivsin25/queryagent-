# Nimbus Analytics — Security and Data Protection Policy

Version 3.2, effective 1 March 2026. Owner: Nimbus Labs Security Team.

## Hosting and data residency

Nimbus Analytics runs on AWS. Customers choose a data region when they create a workspace:
EU (Frankfurt), US (Virginia) or APAC (Singapore). Customer data never leaves the selected region,
except for backups, which are replicated to a second availability zone within the same region.
The region cannot be changed after the workspace is created; moving regions requires a migration
handled by the support team.

## Encryption

All data is encrypted in transit with TLS 1.2 or higher and at rest with AES-256.
Enterprise customers can bring their own encryption keys (BYOK) through AWS KMS.
Encryption keys are rotated automatically every 90 days.

## Access control

- Single sign-on (SSO) with SAML 2.0 and SCIM user provisioning is available on the Enterprise
  plan.
- Two-factor authentication can be enforced for a whole workspace on Business and Enterprise
  plans.
- Nimbus employees cannot access customer data unless the customer grants temporary support
  access, which expires automatically after 72 hours.

## Backups and retention

Backups are taken every 6 hours and kept for 35 days. When a customer deletes their workspace,
all customer data is permanently deleted within 30 days, and backups containing that data
expire within a further 35 days.

## Compliance

Nimbus Analytics holds SOC 2 Type II and ISO/IEC 27001 certifications. The platform is GDPR
compliant and a Data Processing Agreement (DPA) is available to all customers. Nimbus is not
HIPAA compliant and must not be used to store protected health information.

## Security incidents

Customers are notified of a confirmed security incident affecting their data within 48 hours.
Vulnerabilities can be reported to security@nimbus-analytics.example; the bug bounty programme
pays between $250 and $10,000 depending on severity.
