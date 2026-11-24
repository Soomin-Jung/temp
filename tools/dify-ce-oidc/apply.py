#!/usr/bin/env python3
"""Apply CE OIDC overlay to a clean Dify 1.17.1 checkout. No network calls."""
import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("api/libs/ce_oidc.py", "api/controllers/console/auth/ce_oidc.py")
INIT = "api/controllers/console/__init__.py"
WEB = "web/app/signin/normal-form.tsx"

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"Upstream anchor mismatch: {old!r}")
    return text.replace(old, new, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--check", action="store_true", help="validate only")
    args = parser.parse_args()
    root = args.checkout.resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(["git", "-C", str(root), "rev-parse", "1.17.1^{commit}"], text=True).strip()
    if head != tag:
        raise SystemExit("Expected checkout at tag 1.17.1; port and review for other releases.")
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if status:
        raise SystemExit("Expected clean checkout; create a dedicated branch first.")
    if any((root / f).exists() for f in FILES):
        raise SystemExit("Overlay destination already exists.")
    init = (root / INIT).read_text()
    init = replace_once(init, "    activate,\n    data_source_bearer_auth,", "    activate,\n    ce_oidc,\n    data_source_bearer_auth,")
    web = (root / WEB).read_text()
    web = replace_once(web, "import { cn } from '@langgenius/dify-ui/cn'", "import { cn } from '@langgenius/dify-ui/cn'\nimport { API_PREFIX } from '@/config'")
    web = replace_once(web, "  const hasSocialLogin = systemFeatures.enable_social_oauth_login", """  const { data: ceOidcConfig } = useQuery({
    queryKey: ['ce-oidc-config'],
    queryFn: async () => {
      const res = await fetch(API_PREFIX + '/ce-oidc/config', { credentials: 'same-origin', cache: 'no-store' })
      if (!res.ok) return { enabled: false }
      return await res.json() as { enabled: boolean }
    },
    enabled: systemFeatures.deployment_edition === 'COMMUNITY' && !isInviteLink,
    retry: false,
  })
  const hasCeOidc = systemFeatures.deployment_edition === 'COMMUNITY' && !isInviteLink && ceOidcConfig?.enabled === true
  const hasSocialLogin = systemFeatures.enable_social_oauth_login""")
    web = replace_once(web, "(hasSocialLogin || hasSsoLogin) && hasEmailLogin", "(hasSocialLogin || hasSsoLogin || hasCeOidc) && hasEmailLogin")
    web = replace_once(web, "!hasSocialLogin && !hasEmailCodeLogin && !hasEmailPasswordLogin && !hasSsoLogin", "!hasSocialLogin && !hasEmailCodeLogin && !hasEmailPasswordLogin && !hasSsoLogin && !hasCeOidc")
    web = replace_once(web, "{hasSocialLogin && <SocialAuth />}", """{hasSocialLogin && <SocialAuth />}
            {hasCeOidc && (
              <a className="flex w-full items-center justify-center rounded-lg border border-components-button-secondary-border px-3 py-2 text-text-primary"
                href={API_PREFIX + '/ce-oidc/login'}>
                Sign in with your organization (OIDC)
              </a>
            )}""")
    for f in FILES:
        if not (HERE / "overlay" / f).is_file():
            raise SystemExit(f"Missing overlay source: {f}")
    if args.check:
        print("Overlay anchors and target tag verified.")
        return
    for f in FILES:
        shutil.copyfile(HERE / "overlay" / f, root / f)
    (root / INIT).write_text(init)
    (root / WEB).write_text(web)
    print("Applied overlay. Review git diff, run tests and rebuild API/Web.")

if __name__ == "__main__":
    main()
