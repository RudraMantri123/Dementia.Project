#!/usr/bin/env python3
import subprocess
import sys

# Define new commit messages for each commit hash
commit_messages = {
    "480162b": "Prepare testing infrastructure",
    "b2d9ca9": "Add comprehensive technical documentation", 
    "1700722": "Add enhanced user experience features",
    "ecf166a": "Improve ML model accuracy and performance",
    "92e79e0": "Fix backend configuration issues",
    "0bebb39": "Create comprehensive project documentation",
    "2959ae8": "Build responsive frontend interface",
    "fcc7ab9": "Implement real-time analytics dashboard"
}

def rewrite_commits():
    # Get the list of commits
    result = subprocess.run(['git', 'log', '--oneline', '-8'], capture_output=True, text=True)
    commits = result.stdout.strip().split('\n')
    
    for commit_line in commits:
        if not commit_line:
            continue
            
        commit_hash = commit_line.split()[0]
        if commit_hash in commit_messages:
            new_message = commit_messages[commit_hash]
            print(f"Rewriting {commit_hash} to: {new_message}")
            
            # Use git filter-branch to rewrite this specific commit
            subprocess.run([
                'git', 'filter-branch', '-f', '--msg-filter',
                f'if [ "$GIT_COMMIT" = "{commit_hash}" ]; then echo "{new_message}"; else cat; fi',
                '--', 'HEAD~8..HEAD'
            ])

if __name__ == "__main__":
    rewrite_commits()
