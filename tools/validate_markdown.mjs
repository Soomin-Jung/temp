#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const root = path.resolve(process.argv[2] ?? process.cwd());
const errors = [];
const warnings = [];
const stats = {
  files: 0,
  links: 0,
  mathBlocks: 0,
  inlineMathLines: 0,
  mermaidBlocks: 0,
};

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter((entry) => !['.git', 'node_modules'].includes(entry.name))
    .flatMap((entry) => {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) return walk(full);
      return entry.isFile() && entry.name.endsWith('.md') ? [full] : [];
    });
}

function relative(file) {
  return path.relative(root, file).split(path.sep).join('/');
}

function report(bucket, file, line, message) {
  bucket.push(`${relative(file)}:${line}: ${message}`);
}

function stripInlineCode(line) {
  return line.replace(/(`+)(.*?)\1/g, '');
}

function unescapedDollarCount(line) {
  let count = 0;
  for (let i = 0; i < line.length; i += 1) {
    if (line[i] !== '$') continue;
    let slashes = 0;
    for (let j = i - 1; j >= 0 && line[j] === '\\'; j -= 1) slashes += 1;
    if (slashes % 2 === 0) count += 1;
  }
  return count;
}

function validateMathBlock(file, startLine, source) {
  if (!source.trim()) {
    report(errors, file, startLine, 'empty math block');
    return;
  }

  if (source.includes('$$')) {
    report(errors, file, startLine, 'fenced math block must not contain $$ delimiters');
  }

  let braces = 0;
  for (let i = 0; i < source.length; i += 1) {
    if ((source[i] === '{' || source[i] === '}') && source[i - 1] === '\\') continue;
    if (source[i] === '{') braces += 1;
    if (source[i] === '}') braces -= 1;
    if (braces < 0) {
      report(errors, file, startLine, 'closing brace appears before its opening brace');
      break;
    }
  }
  if (braces !== 0) report(errors, file, startLine, `unbalanced braces (${braces})`);

  const envStack = [];
  const envPattern = /\\(begin|end)\{([^}]+)\}/g;
  for (const match of source.matchAll(envPattern)) {
    if (match[1] === 'begin') {
      envStack.push(match[2]);
    } else if (envStack.pop() !== match[2]) {
      report(errors, file, startLine, `mismatched \\end{${match[2]}}`);
    }
  }
  if (envStack.length) {
    report(errors, file, startLine, `unclosed math environment: ${envStack.join(', ')}`);
  }

  const leftCount = (source.match(/\\left(?:\b|[.()\[\]{}|])/g) ?? []).length;
  const rightCount = (source.match(/\\right(?:\b|[.()\[\]{}|])/g) ?? []).length;
  if (leftCount !== rightCount) {
    report(warnings, file, startLine, `\\left / \\right count differs (${leftCount}/${rightCount})`);
  }
}

function resolveLink(file, lineNumber, rawTarget) {
  let target = rawTarget.trim();
  if (target.startsWith('<') && target.endsWith('>')) target = target.slice(1, -1);
  else target = target.split(/\s+["']/)[0];

  if (!target || target.startsWith('#')) return;
  if (/^(?:[a-z][a-z0-9+.-]*:|\/\/|\/)/i.test(target)) return;

  target = target.split('#')[0].split('?')[0];
  try {
    target = decodeURIComponent(target);
  } catch {
    report(errors, file, lineNumber, `invalid URL encoding in link: ${rawTarget}`);
    return;
  }

  const resolved = path.resolve(path.dirname(file), target);
  if (!resolved.startsWith(root + path.sep) && resolved !== root) {
    report(errors, file, lineNumber, `relative link escapes repository: ${rawTarget}`);
    return;
  }

  if (!fs.existsSync(resolved)) {
    report(errors, file, lineNumber, `missing relative link target: ${rawTarget}`);
    return;
  }

  if (fs.statSync(resolved).isDirectory() && !fs.existsSync(path.join(resolved, 'README.md'))) {
    report(warnings, file, lineNumber, `linked directory has no README.md: ${rawTarget}`);
  }
}

for (const file of walk(root).sort()) {
  stats.files += 1;
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  let fence = null;
  let mathStart = 0;
  let mathLines = [];
  let mermaidStart = 0;
  let mermaidLines = [];

  for (let index = 0; index < lines.length; index += 1) {
    const lineNumber = index + 1;
    const line = lines[index];
    const marker = line.match(/^\s*(`{3,}|~{3,})(.*)$/);

    if (marker) {
      const char = marker[1][0];
      const length = marker[1].length;
      const info = marker[2].trim().toLowerCase();

      if (!fence) {
        fence = { char, length, info, line: lineNumber };
        if (info === 'math') {
          mathStart = lineNumber;
          mathLines = [];
          stats.mathBlocks += 1;
        } else if (info === 'mermaid') {
          mermaidStart = lineNumber;
          mermaidLines = [];
          stats.mermaidBlocks += 1;
        }
        continue;
      }

      if (char === fence.char && length >= fence.length && !info) {
        if (fence.info === 'math') validateMathBlock(file, mathStart, mathLines.join('\n'));
        if (fence.info === 'mermaid' && !mermaidLines.join('\n').trim()) {
          report(errors, file, mermaidStart, 'empty Mermaid block');
        }
        fence = null;
        continue;
      }
    }

    if (fence) {
      if (fence.info === 'math') mathLines.push(line);
      if (fence.info === 'mermaid') mermaidLines.push(line);
      continue;
    }

    const prose = stripInlineCode(line);
    if (prose.includes('$$')) report(errors, file, lineNumber, 'raw $$ delimiter remains outside a code fence');

    const dollars = unescapedDollarCount(prose);
    if (dollars % 2 !== 0) report(errors, file, lineNumber, `unmatched inline math delimiter ($ count: ${dollars})`);
    if (dollars >= 2) stats.inlineMathLines += 1;

    const linkPattern = /!?\[[^\]]*\]\(([^)]+)\)/g;
    for (const match of prose.matchAll(linkPattern)) {
      stats.links += 1;
      resolveLink(file, lineNumber, match[1]);
    }
  }

  if (fence) report(errors, file, fence.line, `unclosed ${fence.info || 'code'} fence`);
}

console.log(`Markdown files: ${stats.files}`);
console.log(`Relative/external links parsed: ${stats.links}`);
console.log(`Display-math blocks: ${stats.mathBlocks}`);
console.log(`Lines with inline math: ${stats.inlineMathLines}`);
console.log(`Mermaid blocks: ${stats.mermaidBlocks}`);

if (warnings.length) {
  console.log(`\nWarnings (${warnings.length}):`);
  for (const warning of warnings) console.log(`- ${warning}`);
}

if (errors.length) {
  console.error(`\nErrors (${errors.length}):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exitCode = 1;
} else {
  console.log('\nOK: Markdown structure, math delimiters, math-block balance, and relative links passed.');
}
