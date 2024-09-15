while read repo; do
  gh repo delete "$repo" --yes
done < repos.txt
