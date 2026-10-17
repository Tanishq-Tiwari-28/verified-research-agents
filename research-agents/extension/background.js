// Side panel (not a popup) so a long-running question isn't cancelled when focus moves.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(console.error);
