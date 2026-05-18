-- ========================================================================== --
-- ==                           BASIC SETTINGS                             == --
-- ========================================================================== --

local opt = vim.opt

opt.backup = false
opt.clipboard = "unnamedplus"
opt.cmdheight = 1
opt.completeopt = { "menuone", "noselect" }
opt.expandtab = true
opt.fileencoding = "utf-8"
opt.hlsearch = true
opt.ignorecase = true
opt.smartcase = true
opt.mouse = "a"
opt.number = true
opt.relativenumber = false
opt.scrolloff = 8
opt.shiftwidth = 2
opt.tabstop = 2
opt.smartindent = true
opt.signcolumn = "yes"
opt.termguicolors = true
opt.undofile = true
opt.updatetime = 250
opt.wrap = false

vim.g.mapleader = " "

local keymap = vim.keymap.set
keymap("i", "jj", "<ESC>", { desc = "Exit insert mode" })

-- ========================================================================== --
-- ==                              PLUGINS                                  == --
-- ========================================================================== --

local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.uv.fs_stat(lazypath) then
  vim.fn.system({
    "git", "clone", "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git", "--branch=stable", lazypath,
  })
end
vim.opt.rtp:prepend(lazypath)

require("lazy").setup({
  -- Colorscheme
  { "embark-theme/vim", name = "embark", lazy = false, priority = 1000,
    config = function() vim.cmd([[colorscheme embark]]) end },

  -- UI
  { "nvim-lualine/lualine.nvim", opts = { options = { theme = "embark" } } },
  { "nvim-tree/nvim-web-devicons", lazy = true },

  -- Navigation
  { "tpope/vim-surround" },
  { "ThePrimeagen/harpoon", branch = "harpoon2", dependencies = { "nvim-lua/plenary.nvim" } },
  { "nvim-telescope/telescope.nvim", dependencies = { "nvim-lua/plenary.nvim" } },
  { "nvim-tree/nvim-tree.lua", opts = {} },

  -- Syntax
  { "nvim-treesitter/nvim-treesitter", build = ":TSUpdate" },
})

-- ========================================================================== --
-- ==                              KEYMAPS                                  == --
-- ========================================================================== --

-- Telescope
local builtin = require("telescope.builtin")
keymap("n", "<leader>ff", builtin.find_files, { desc = "Find Files" })
keymap("n", "<leader>fg", builtin.live_grep,  { desc = "Grep Project" })
keymap("n", "<leader>fb", builtin.buffers,    { desc = "Buffers" })

-- nvim-tree
keymap("n", "<leader>e", ":NvimTreeToggle<CR>", { desc = "Toggle File Tree" })

-- Harpoon
local harpoon = require("harpoon")
harpoon:setup()
keymap("n", "<leader>a", function() harpoon:list():add() end,                        { desc = "Harpoon Add" })
keymap("n", "<C-e>",     function() harpoon.ui:toggle_quick_menu(harpoon:list()) end, { desc = "Harpoon Menu" })
keymap("n", "<C-h>",     function() harpoon:list():select(1) end, { desc = "Harpoon 1" })
keymap("n", "<C-j>",     function() harpoon:list():select(2) end, { desc = "Harpoon 2" })
keymap("n", "<C-k>",     function() harpoon:list():select(3) end, { desc = "Harpoon 3" })
keymap("n", "<C-l>",     function() harpoon:list():select(4) end, { desc = "Harpoon 4" })
